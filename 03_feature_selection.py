# 檔名: 03_feature_selection.py
# 描述: 從龐大的特徵集中，利用 LightGBM 模型找出對預測目標最重要的特徵。
# 版本: 2.0 (與 04 號腳本同步，使用三道門檻標籤法)

import logging
import sys
import json
import yaml  # <--- 新增
from pathlib import Path
from typing import List, Dict

import pandas as pd
import lightgbm as lgb

# ==============================================================================
# 1. 複製三道門檻標籤函式 (從 04_parameter_optimization.py 複製過來)
# ==============================================================================
def create_triple_barrier_labels(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """
    為 DataFrame 創建三道門檻標籤。
    """
    df_out = df.copy()
    tp_pct = settings['tp_pct']
    sl_pct = settings['sl_pct']
    max_hold = settings['max_hold_periods']
    
    outcomes = pd.DataFrame(index=df_out.index, columns=['hit_time', 'label'])
    
    for i in range(len(df_out) - max_hold):
        entry_price = df_out['close'].iloc[i]
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
        
        window = df_out.iloc[i+1 : i+1+max_hold]
        
        hit_tp_time = window[window['high'] >= tp_price].index.min()
        hit_sl_time = window[window['low'] <= sl_price].index.min()
        
        if pd.notna(hit_tp_time) and pd.notna(hit_sl_time):
            if hit_tp_time < hit_sl_time:
                outcomes.loc[df_out.index[i], 'label'] = 1
                outcomes.loc[df_out.index[i], 'hit_time'] = hit_tp_time
            else:
                outcomes.loc[df_out.index[i], 'label'] = -1
                outcomes.loc[df_out.index[i], 'hit_time'] = hit_sl_time
        elif pd.notna(hit_tp_time):
            outcomes.loc[df_out.index[i], 'label'] = 1
            outcomes.loc[df_out.index[i], 'hit_time'] = hit_tp_time
        elif pd.notna(hit_sl_time):
            outcomes.loc[df_out.index[i], 'label'] = -1
            outcomes.loc[df_out.index[i], 'hit_time'] = hit_sl_time
        else:
            outcomes.loc[df_out.index[i], 'label'] = 0
            outcomes.loc[df_out.index[i], 'hit_time'] = window.index[-1]
            
    df_out = df_out.join(outcomes)
    # 關鍵：目標與 04 號腳本的原始定義保持一致
    df_out['target'] = (df_out['label'] == 1).astype(int)
    return df_out

# ==============================================================================
# 2. 配置區塊
# ==============================================================================
class Config:
    """儲存腳本所需的所有配置參數。"""
    INPUT_BASE_DIR = Path("Output_Feature_Engineering/MarketData_with_Combined_Features_v3")
    OUTPUT_BASE_DIR = Path("Output_ML_Pipeline")
    OUTPUT_FILENAME = "selected_features.json"
    CONFIG_FILE_PATH = Path("config.yaml") # <--- 新增：指向主設定檔

    # --- 模型與特徵篩選相關參數 ---
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
        'verbose': -1,
    }
    LOG_LEVEL = "INFO"

# ==============================================================================
# 3. 特徵篩選器類別 (修改版)
# ==============================================================================
class FeatureSelector:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        # 載入主設定檔以獲取三道門檻設定
        try:
            with open(self.config.CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
            self.tb_settings = full_config['triple_barrier_settings']
            self.logger.info(f"成功從 {self.config.CONFIG_FILE_PATH} 載入三道門檻設定。")
        except Exception as e:
            self.logger.critical(f"讀取設定檔 {self.config.CONFIG_FILE_PATH} 失敗: {e}")
            sys.exit(1)

    def _setup_logger(self) -> logging.Logger:
        # ... 此函式內容不變 ...
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.LOG_LEVEL.upper())
        if logger.hasHandlers():
            logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        return logger

    def find_input_files(self) -> List[Path]:
        # ... 此函式內容不變 ...
        self.logger.info(f"從 '{self.config.INPUT_BASE_DIR}' 尋找輸入檔案...")
        files = list(self.config.INPUT_BASE_DIR.rglob("*.parquet"))
        self.logger.info(f"找到了 {len(files)} 個 Parquet 檔案。")
        return files

    def get_feature_importance_for_file(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        為單一 DataFrame 計算特徵重要性 (已更新為使用三道門檻標籤)。
        """
        # 1. 準備數據
        self.logger.debug("準備特徵 (X) 和目標 (y)...")
        
        non_feature_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'target', 'time', 'spread', 'real_volume', 'label', 'hit_time']
        features = [col for col in df.columns if col not in non_feature_cols]
        
        # ★★★ 使用新的標籤函式 ★★★
        df_labeled = create_triple_barrier_labels(df, self.tb_settings)
        
        # 確保特徵欄位在 df_labeled 中仍然存在
        features = [f for f in features if f in df_labeled.columns]
        
        X = df_labeled[features]
        y = df_labeled['target']
        
        combined = pd.concat([X, y], axis=1)
        combined.dropna(inplace=True)
        
        X = combined[features]
        y = combined['target']

        if len(X) == 0:
            self.logger.warning("數據清洗後沒有剩餘樣本，無法進行特徵篩選。")
            return pd.DataFrame({'feature': [], 'importance': []})

        # 2. 訓練模型
        self.logger.debug(f"開始使用 LightGBM 訓練模型... 樣本數: {len(X)}")
        model = lgb.LGBMClassifier(**self.config.LGBM_PARAMS)
        model.fit(X, y)

        # 3. 提取特徵重要性
        feature_importances = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        })
        
        return feature_importances

    def save_selected_features(self, features: List[str]) -> None:
        # ... 此函式內容不變 ...
        output_path = self.config.OUTPUT_BASE_DIR / self.config.OUTPUT_FILENAME
        self.logger.info(f"正在將選出的特徵儲存到: {output_path}")
        
        output_data = {
            "description": "由 03_feature_selection.py (v2.0 - TBM) 產生的全域最佳特徵列表",
            "feature_count": len(features),
            "selected_features": features
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        
        self.logger.info("特徵列表儲存成功。")

    def run(self) -> None:
        # ... 此函式內容大部分不變，只需確保日誌訊息更新 ...
        self.logger.info("========= 特徵篩選流程開始 (v2.0 - 三道門檻標籤模式) =========")
        input_files = self.find_input_files()
        
        if not input_files:
            self.logger.warning("在輸入目錄中沒有找到任何檔案，流程結束。")
            return

        all_importances = []
        for file_path in input_files:
            try:
                self.logger.info(f"--- 正在處理檔案: {file_path.name} ---")
                df = pd.read_parquet(file_path)
                
                importance_df = self.get_feature_importance_for_file(df)
                if not importance_df.empty:
                    all_importances.append(importance_df)

            except Exception as e:
                self.logger.error(f"處理檔案 {file_path.name} 時發生錯誤: {e}", exc_info=True)
        
        if not all_importances:
            self.logger.error("未能從任何檔案中成功計算特徵重要性，流程終止。")
            return
            
        self.logger.info("正在彙總所有檔案的特徵重要性...")
        global_importance = pd.concat(all_importances).groupby('feature')['importance'].sum().sort_values(ascending=False)
        
        self.logger.info("\n--- 全域特徵重要性排名 (前 30) ---\n" + global_importance.head(30).to_string())

        top_features = global_importance.head(self.config.TOP_N_FEATURES).index.tolist()
        self.logger.info(f"\n--- 已選出全域最重要的 {self.config.TOP_N_FEATURES} 個特徵 ---")
        for f in top_features:
            self.logger.info(f"- {f}")
            
        self.save_selected_features(top_features)
            
        self.logger.info("========= 特徵篩選流程結束 =========")


if __name__ == "__main__":
    try:
        config = Config()
        selector = FeatureSelector(config)
        selector.run()
    except Exception as e:
        logging.critical(f"特徵篩選腳本執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
        sys.exit(1)
