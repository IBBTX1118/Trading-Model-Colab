# 檔名: 03_feature_selection.py
# 描述: 為每一個市場單獨計算並儲存其最重要的特徵。
# 版本: 4.0 (市場專屬特徵篩選)

import logging
import sys
import json
import yaml
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import pandas as pd
import lightgbm as lgb

# ==============================================================================
# 1. 輔助函式 (與 04 號腳本同步)
# ==============================================================================
def create_triple_barrier_labels(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """為 DataFrame 創建基於 ATR 的動態三道門檻標籤。"""
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
        if atr_at_entry <= 0 or pd.isna(atr_at_entry): continue
        tp_price = entry_price + (atr_at_entry * tp_multiplier); sl_price = entry_price - (atr_at_entry * sl_multiplier)
        window = df_out.iloc[i+1 : i+1+max_hold]
        hit_tp_time = window[high_series.iloc[i+1:i+1+max_hold] >= tp_price].index.min()
        hit_sl_time = window[low_series.iloc[i+1:i+1+max_hold] <= sl_price].index.min()
        if pd.notna(hit_tp_time) and pd.notna(hit_sl_time):
            outcomes.loc[df_out.index[i], 'label'] = 1 if hit_tp_time < hit_sl_time else -1
        elif pd.notna(hit_tp_time): outcomes.loc[df_out.index[i], 'label'] = 1
        elif pd.notna(hit_sl_time): outcomes.loc[df_out.index[i], 'label'] = -1
        else: outcomes.loc[df_out.index[i], 'label'] = 0
    df_out = df_out.join(outcomes); df_out['target'] = (df_out['label'] == 1).astype(int); return df_out

# ==============================================================================
# 2. 配置區塊
# ==============================================================================
class Config:
    INPUT_BASE_DIR = Path("Output_Feature_Engineering/MarketData_with_Combined_Features_v3")
    OUTPUT_BASE_DIR = Path("Output_ML_Pipeline")
    # 檔名將動態生成，例如 "selected_features_AUDUSD_sml_H4.json"
    CONFIG_FILE_PATH = Path("config.yaml")
    TOP_N_FEATURES: int = 20
    LGBM_PARAMS = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1, 'seed': 42, 'n_jobs': -1, 'verbose': -1}
    LOG_LEVEL = "INFO"

# ==============================================================================
# 3. 特徵篩選器類別 (市場專屬版)
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
            self.logger.critical(f"讀取設定檔 {self.config.CONFIG_FILE_PATH} 失敗: {e}"); sys.exit(1)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.LOG_LEVEL.upper())
        if logger.hasHandlers(): logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(formatter); logger.addHandler(sh)
        return logger

    def get_feature_importance_for_file(self, df: pd.DataFrame) -> pd.DataFrame:
        non_feature_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'target', 'time', 'spread', 'real_volume', 'label', 'hit_time']
        df_labeled = create_triple_barrier_labels(df, self.tb_settings)
        features = [col for col in df_labeled.columns if col not in non_feature_cols]
        X = df_labeled[features]; y = df_labeled['target']
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[features]; y = combined['target']
        if len(X) == 0: return pd.DataFrame({'feature': [], 'importance': []})
        model = lgb.LGBMClassifier(**self.config.LGBM_PARAMS); model.fit(X, y)
        return pd.DataFrame({'feature': features, 'importance': model.feature_importances_})

    def save_selected_features(self, features: List[str], market_name: str) -> None:
        """將選出的特徵列表儲存為市場專屬的 JSON 檔案。"""
        output_filename = f"selected_features_{market_name}.json"
        output_path = self.config.OUTPUT_BASE_DIR / output_filename
        self.logger.info(f"正在將 {market_name} 的特徵儲存到: {output_path}")
        output_data = {
            "description": f"由 03_feature_selection.py (v4.0) 為市場 {market_name} 產生的專屬特徵列表",
            "market": market_name, "feature_count": len(features), "selected_features": features
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        self.logger.info(f"{market_name} 的特徵列表儲存成功。")

    def run(self) -> None:
        """執行完整特徵篩選流程，為每個市場生成獨立的特徵文件。"""
        self.logger.info("========= 特徵篩選流程開始 (v4.0 - 市場專屬模式) =========")
        all_files = list(self.config.INPUT_BASE_DIR.rglob("*.parquet"))
        if not all_files: self.logger.warning("在輸入目錄中沒有找到任何檔案，流程結束。"); return
        
        # ★★★ 關鍵修改：按市場對檔案進行分組 ★★★
        market_files = defaultdict(list)
        for f in all_files:
            market_name = f.stem  # e.g., "AUDUSD_sml_H4"
            market_files[market_name].append(f)
        
        self.logger.info(f"發現 {len(market_files)} 個市場組需要處理: {list(market_files.keys())}")

        for market_name, files in market_files.items():
            self.logger.info(f"\n{'='*20} 開始處理市場: {market_name} {'='*20}")
            all_importances = []
            for file_path in files:
                try:
                    self.logger.info(f"--- 正在讀取檔案: {file_path.name} ---")
                    df = pd.read_parquet(file_path)
                    importance_df = self.get_feature_importance_for_file(df)
                    if not importance_df.empty: all_importances.append(importance_df)
                except Exception as e:
                    self.logger.error(f"處理檔案 {file_path.name} 時發生錯誤: {e}", exc_info=True)
            
            if not all_importances:
                self.logger.warning(f"市場 {market_name} 未能成功計算任何特徵重要性，已跳過。"); continue
            
            market_importance = pd.concat(all_importances).groupby('feature')['importance'].sum().sort_values(ascending=False)
            self.logger.info(f"\n--- {market_name} 特徵重要性排名 (前 30) ---\n" + market_importance.head(30).to_string())
            top_features = market_importance.head(self.config.TOP_N_FEATURES).index.tolist()
            self.logger.info(f"\n--- 為 {market_name} 選出最重要的 {self.config.TOP_N_FEATURES} 個特徵 ---")
            for f in top_features: self.logger.info(f"- {f}")
            self.save_selected_features(top_features, market_name)
            
        self.logger.info("\n========= 所有市場的專屬特徵篩選流程結束 =========")

if __name__ == "__main__":
    try:
        config = Config(); selector = FeatureSelector(config); selector.run()
    except Exception as e:
        logging.critical(f"特徵篩選腳本執行時發生未預期的嚴重錯誤: {e}", exc_info=True); sys.exit(1)
