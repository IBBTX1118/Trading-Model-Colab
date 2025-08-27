# 檔名: 03_feature_selection.py
# 描述: 為每一個市場單獨計算並儲存其最重要的特徵。
# 版本: 4.1 (市場專屬特徵篩選 + 自適應標籤創建)

import logging
import sys
import json
import yaml
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import pandas as pd
import numpy as np
import lightgbm as lgb

# ==============================================================================
# 1. 輔助函式 (更新版本，加入自適應標籤創建)
# ==============================================================================
def create_adaptive_labels(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """自適應標籤創建，根據市場狀態動態調整止盈止損"""
    df_out = df.copy()
    
    # 基礎倍數
    tp_multiplier_base = settings['tp_atr_multiplier']
    sl_multiplier_base = settings['sl_atr_multiplier']
    max_hold = settings['max_hold_periods']
    
    # 狀態調整因子 [cite: 71]
    regime_adjustment = {
        0: 0.8,  # 低波動盤整: 收緊目標 [cite: 73]
        1: 0.9,  # 低波動趨勢: 略微收緊 [cite: 74]
        2: 1.1,  # 高波動盤整: 略微放大 [cite: 75]
        3: 1.2   # 高波動趨勢: 放大目標 [cite: 76]
    }
    
    # 檢查 market_regime 是否存在
    if 'market_regime' in df_out.columns:
        df_out['tp_multiplier_adj'] = df_out['market_regime'].map(regime_adjustment) * tp_multiplier_base
        df_out['sl_multiplier_adj'] = df_out['market_regime'].map(regime_adjustment) * sl_multiplier_base
        logging.info("使用市場狀態自適應調整止盈止損倍數")
    else:
        # 如果沒有狀態特徵，則退回使用固定倍數
        df_out['tp_multiplier_adj'] = tp_multiplier_base
        df_out['sl_multiplier_adj'] = sl_multiplier_base
        logging.warning("未找到 market_regime 特徵，使用固定止盈止損倍數")
    
    # 檢查ATR欄位
    atr_col_name = 'D1_ATR_14' if 'D1_ATR_14' in df_out.columns else 'ATR_14'
    if atr_col_name not in df_out.columns:
        raise ValueError(f"數據中缺少 ATR 欄位 ('ATR_14' 或 'D1_ATR_14')，無法創建標籤。")
    
    # 創建標籤結果DataFrame
    outcomes = pd.DataFrame(index=df_out.index, columns=['label'])
    
    high_series, low_series, atr_series = df_out['high'], df_out['low'], df_out[atr_col_name]
    tp_multipliers, sl_multipliers = df_out['tp_multiplier_adj'], df_out['sl_multiplier_adj']
    
    # 為每個時間點計算三道門檻標籤
    for i in range(len(df_out) - max_hold):
        entry_price = df_out['close'].iloc[i]
        atr_at_entry = atr_series.iloc[i]
        tp_multiplier = tp_multipliers.iloc[i]
        sl_multiplier = sl_multipliers.iloc[i]
        
        # 檢查數據有效性
        if atr_at_entry <= 0 or pd.isna(atr_at_entry) or pd.isna(tp_multiplier) or pd.isna(sl_multiplier):
            continue
            
        # 計算自適應的止盈止損價格
        tp_price = entry_price + (atr_at_entry * tp_multiplier)
        sl_price = entry_price - (atr_at_entry * sl_multiplier)
        
        # 獲取後續窗口數據
        window_high = high_series.iloc[i+1:i+1+max_hold]
        window_low = low_series.iloc[i+1:i+1+max_hold]
        
        # 找到觸發止盈和止損的時間點
        hit_tp_mask = window_high >= tp_price
        hit_sl_mask = window_low <= sl_price
        
        hit_tp_time = window_high[hit_tp_mask].index.min() if hit_tp_mask.any() else pd.NaT
        hit_sl_time = window_low[hit_sl_mask].index.min() if hit_sl_mask.any() else pd.NaT
        
        # 判斷標籤結果
        if pd.notna(hit_tp_time) and pd.notna(hit_sl_time):
            # 兩者都觸發，看哪個先發生
            outcomes.loc[df_out.index[i], 'label'] = 1 if hit_tp_time < hit_sl_time else -1
        elif pd.notna(hit_tp_time):
            # 只觸發止盈
            outcomes.loc[df_out.index[i], 'label'] = 1
        elif pd.notna(hit_sl_time):
            # 只觸發止損
            outcomes.loc[df_out.index[i], 'label'] = -1
        else:
            # 都沒觸發
            outcomes.loc[df_out.index[i], 'label'] = 0
    
    # 合併結果並創建二元目標變數
    df_out = df_out.join(outcomes)
    df_out['target'] = (df_out['label'] == 1).astype(int)
    
    # 清理臨時欄位
    df_out.drop(columns=['tp_multiplier_adj', 'sl_multiplier_adj'], inplace=True, errors='ignore')
    
    return df_out

def create_triple_barrier_labels(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """為 DataFrame 創建基於 ATR 的動態三道門檻標籤。(保留為向後兼容)"""
    df_out = df.copy()
    tp_multiplier = settings['tp_atr_multiplier']
    sl_multiplier = settings['sl_atr_multiplier']
    max_hold = settings['max_hold_periods']
    
    atr_col_name = 'D1_ATR_14' if 'D1_ATR_14' in df_out.columns else 'ATR_14'
    if atr_col_name not in df_out.columns:
        raise ValueError(f"數據中缺少 ATR 欄位 ('ATR_14' 或 'D1_ATR_14')，無法創建標籤。")

    outcomes = pd.DataFrame(index=df_out.index, columns=['label'])
    
    high_series, low_series, atr_series = df_out['high'], df_out['low'], df_out[atr_col_name]
    
    for i in range(len(df_out) - max_hold):
        entry_price, atr_at_entry = df_out['close'].iloc[i], atr_series.iloc[i]
        if atr_at_entry <= 0 or pd.isna(atr_at_entry): 
            continue
        tp_price = entry_price + (atr_at_entry * tp_multiplier)
        sl_price = entry_price - (atr_at_entry * sl_multiplier)
        window = df_out.iloc[i+1 : i+1+max_hold]
        hit_tp_time = window[high_series.iloc[i+1:i+1+max_hold] >= tp_price].index.min()
        hit_sl_time = window[low_series.iloc[i+1:i+1+max_hold] <= sl_price].index.min()
        if pd.notna(hit_tp_time) and pd.notna(hit_sl_time):
            outcomes.loc[df_out.index[i], 'label'] = 1 if hit_tp_time < hit_sl_time else -1
        elif pd.notna(hit_tp_time): 
            outcomes.loc[df_out.index[i], 'label'] = 1
        elif pd.notna(hit_sl_time): 
            outcomes.loc[df_out.index[i], 'label'] = -1
        else: 
            outcomes.loc[df_out.index[i], 'label'] = 0
    df_out = df_out.join(outcomes)
    df_out['target'] = (df_out['label'] == 1).astype(int)
    return df_out

# ==============================================================================
# 2. 配置區塊
# ==============================================================================
class Config:
    INPUT_BASE_DIR = Path("Output_Feature_Engineering/MarketData_with_Combined_Features_v3")
    OUTPUT_BASE_DIR = Path("Output_ML_Pipeline")
    # 檔名將動態生成，例如 "selected_features_AUDUSD_sml_H4.json"
    CONFIG_FILE_PATH = Path("config.yaml")
    TOP_N_FEATURES: int = 20
    LGBM_PARAMS = {
        'objective': 'binary', 
        'metric': 'binary_logloss', 
        'boosting_type': 'gbdt', 
        'n_estimators': 200, 
        'learning_rate': 0.05, 
        'num_leaves': 31, 
        'max_depth': -1, 
        'seed': 42, 
        'n_jobs': -1, 
        'verbose': -1
    }
    LOG_LEVEL = "INFO"
    USE_ADAPTIVE_LABELS = True  # 新增：是否使用自適應標籤

# ==============================================================================
# 3. 特徵篩選器類別 (市場專屬版 + 自適應標籤)
# ==============================================================================
class FeatureSelector:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config.CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
            self.tb_settings = full_config['triple_barrier_settings']
            self.logger.info(f"成功從 {self.config.CONFIG_FILE_PATH} 載入三道門檻設定。")
        except Exception as e:
            self.logger.critical(f"讀取設定檔 {self.config.CONFIG_FILE_PATH} 失敗: {e}")
            sys.exit(1)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.LOG_LEVEL.upper())
        if logger.hasHandlers(): 
            logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        return logger

    def get_feature_importance_for_file(self, df: pd.DataFrame) -> pd.DataFrame:
        """為單個檔案計算特徵重要性，支援自適應標籤創建"""
        non_feature_cols = [
            'open', 'high', 'low', 'close', 'tick_volume', 'target', 
            'time', 'spread', 'real_volume', 'label', 'hit_time'
        ]
        
        # ★★★ 使用自適應標籤創建或傳統方法 ★★★
        if self.config.USE_ADAPTIVE_LABELS:
            self.logger.debug("使用自適應標籤創建方法")
            df_labeled = create_adaptive_labels(df, self.tb_settings)
        else:
            self.logger.debug("使用傳統固定倍數標籤創建方法")
            df_labeled = create_triple_barrier_labels(df, self.tb_settings)
        
        # 準備特徵和目標變數
        features = [col for col in df_labeled.columns if col not in non_feature_cols]
        X = df_labeled[features]
        y = df_labeled['target']
        
        # 移除包含NaN的行
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[features]
        y = combined['target']
        
        if len(X) == 0:
            self.logger.warning("處理後的數據為空，返回空的特徵重要性")
            return pd.DataFrame({'feature': [], 'importance': []})
        
        # 檢查目標變數的分佈
        target_distribution = y.value_counts()
        self.logger.debug(f"目標變數分佈: {target_distribution.to_dict()}")
        
        # 如果只有一個類別，無法進行分類
        if len(target_distribution) < 2:
            self.logger.warning("目標變數只有一個類別，無法進行特徵重要性計算")
            return pd.DataFrame({'feature': [], 'importance': []})
        
        # 訓練模型並獲取特徵重要性
        try:
            model = lgb.LGBMClassifier(**self.config.LGBM_PARAMS)
            model.fit(X, y)
            importance_df = pd.DataFrame({
                'feature': features, 
                'importance': model.feature_importances_
            })
            return importance_df
        except Exception as e:
            self.logger.error(f"模型訓練失敗: {e}")
            return pd.DataFrame({'feature': [], 'importance': []})

    def save_selected_features(self, features: List[str], market_name: str) -> None:
        """將選出的特徵列表儲存為市場專屬的 JSON 檔案。"""
        output_filename = f"selected_features_{market_name}.json"
        output_path = self.config.OUTPUT_BASE_DIR / output_filename
        self.logger.info(f"正在將 {market_name} 的特徵儲存到: {output_path}")
        
        output_data = {
            "description": f"由 03_feature_selection.py (v4.1) 為市場 {market_name} 產生的專屬特徵列表",
            "market": market_name, 
            "feature_count": len(features), 
            "selected_features": features,
            "adaptive_labels_used": self.config.USE_ADAPTIVE_LABELS,
            "generation_settings": {
                "top_n_features": self.config.TOP_N_FEATURES,
                "triple_barrier_settings": self.tb_settings
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        self.logger.info(f"{market_name} 的特徵列表儲存成功。")

    def run(self) -> None:
        """執行完整特徵篩選流程，為每個市場生成獨立的特徵文件。"""
        label_method = "自適應標籤" if self.config.USE_ADAPTIVE_LABELS else "固定倍數標籤"
        self.logger.info(f"========= 特徵篩選流程開始 (v4.1 - 市場專屬模式，{label_method}) =========")
        
        all_files = list(self.config.INPUT_BASE_DIR.rglob("*.parquet"))
        if not all_files: 
            self.logger.warning("在輸入目錄中沒有找到任何檔案，流程結束。")
            return
        
        # ★★★ 關鍵修改：按市場對檔案進行分組 ★★★
        market_files = defaultdict(list)
        for f in all_files:
            market_name = f.stem  # e.g., "AUDUSD_sml_H4"
            market_files[market_name].append(f)
        
        self.logger.info(f"發現 {len(market_files)} 個市場組需要處理: {list(market_files.keys())}")

        successful_markets = 0
        total_markets = len(market_files)

        for market_name, files in market_files.items():
            self.logger.info(f"\n{'='*20} 開始處理市場: {market_name} {'='*20}")
            all_importances = []
            
            for file_path in files:
                try:
                    self.logger.info(f"--- 正在讀取檔案: {file_path.name} ---")
                    df = pd.read_parquet(file_path)
                    
                    # 檢查數據基本信息
                    self.logger.debug(f"數據形狀: {df.shape}")
                    self.logger.debug(f"是否包含 market_regime: {'market_regime' in df.columns}")
                    
                    importance_df = self.get_feature_importance_for_file(df)
                    if not importance_df.empty: 
                        all_importances.append(importance_df)
                        self.logger.debug(f"成功計算 {len(importance_df)} 個特徵的重要性")
                    else:
                        self.logger.warning(f"檔案 {file_path.name} 未能產生有效的特徵重要性")
                        
                except Exception as e:
                    self.logger.error(f"處理檔案 {file_path.name} 時發生錯誤: {e}", exc_info=True)
            
            if not all_importances:
                self.logger.warning(f"市場 {market_name} 未能成功計算任何特徵重要性，已跳過。")
                continue
            
            # 聚合所有檔案的特徵重要性
            market_importance = pd.concat(all_importances).groupby('feature')['importance'].sum().sort_values(ascending=False)
            
            self.logger.info(f"\n--- {market_name} 特徵重要性排名 (前 30) ---\n" + market_importance.head(30).to_string())
            
            # 選擇Top N特徵
            top_features = market_importance.head(self.config.TOP_N_FEATURES).index.tolist()
            
            self.logger.info(f"\n--- 為 {market_name} 選出最重要的 {self.config.TOP_N_FEATURES} 個特徵 ---")
            for i, feature in enumerate(top_features, 1):
                importance_score = market_importance[feature]
                self.logger.info(f"{i:2d}. {feature} (重要性: {importance_score:.4f})")
            
            # 儲存特徵列表
            self.save_selected_features(top_features, market_name)
            successful_markets += 1
            
        self.logger.info(f"\n========= 特徵篩選流程完成 =========")
        self.logger.info(f"成功處理 {successful_markets}/{total_markets} 個市場")
        self.logger.info(f"使用方法: {label_method}")

if __name__ == "__main__":
    try:
        config = Config()
        selector = FeatureSelector(config)
        selector.run()
    except Exception as e:
        logging.critical(f"特徵篩選腳本執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
        sys.exit(1)
