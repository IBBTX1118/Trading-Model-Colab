# 檔名: 03_feature_selection.py
# 描述: 從龐大的特徵集中，利用 LightGBM 模型找出對預測目標最重要的特徵。
# 版本: 1.0

import logging
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==============================================================================
# 1. 配置區塊
# ==============================================================================
class Config:
    """儲存腳本所需的所有配置參數。"""
    # 輸入目錄：來自 02_feature_engineering.py 的輸出
    INPUT_BASE_DIR = Path("Output_Feature_Engineering/MarketData_with_All_Features")
    
    # 輸出目錄：儲存機器學習流程的產出
    OUTPUT_BASE_DIR = Path("Output_ML_Pipeline")
    OUTPUT_FILENAME = "selected_features.json"

    # --- Labeling (目標定義) 相關參數 ---
    # 預測未來幾根 K 棒
    LABEL_LOOK_FORWARD_PERIODS: int = 5
    # 目標報酬率閾值 (例如：0.5% -> 0.005)
    LABEL_RETURN_THRESHOLD: float = 0.005

    # --- 模型與特徵篩選相關參數 ---
    # 要選擇最重要的前 N 個特徵
    TOP_N_FEATURES: int = 20
    # LightGBM 模型參數
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
# 2. 特徵篩選器類別
# ==============================================================================
class FeatureSelector:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """設定日誌記錄器。"""
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
        """尋找輸入的 Parquet 檔案。"""
        self.logger.info(f"從 '{self.config.INPUT_BASE_DIR}' 尋找輸入檔案...")
        files = list(self.config.INPUT_BASE_DIR.rglob("*.parquet"))
        self.logger.info(f"找到了 {len(files)} 個 Parquet 檔案。")
        return files

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        定義並創建預測目標 (Label)。
        目標：預測未來 N 根 K 棒後的收盤價是否上漲超過一個閾值。
        - 1: 上漲超過閾值 (我們想買入的訊號)
        - 0: 未上漲超過閾值
        """
        self.logger.info(f"正在創建預測目標 (Label)... 向前看 {self.config.LABEL_LOOK_FORWARD_PERIODS} 根 K 棒，閾值 {self.config.LABEL_RETURN_THRESHOLD:.2%}")
        
        # 計算未來 N 期後的報酬率
        future_returns = df['close'].shift(-self.config.LABEL_LOOK_FORWARD_PERIODS) / df['close'] - 1
        
        # 根據報酬率是否超過閾值來設定目標
        df['target'] = (future_returns > self.config.LABEL_RETURN_THRESHOLD).astype(int)
        
        return df

    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        使用 LightGBM 模型訓練並選出最重要的特徵。
        """
        # 1. 準備數據
        self.logger.info("準備特徵 (X) 和目標 (y)...")
        
        # 排除非特徵欄位
        non_feature_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'target', 'time']
        features = [col for col in df.columns if col not in non_feature_cols]
        
        X = df[features]
        y = df['target']
        
        # 由於 shift 操作和指標計算會產生 NaN，需要移除
        combined = pd.concat([X, y], axis=1)
        combined.dropna(inplace=True)
        
        X = combined[features]
        y = combined['target']

        if len(X) == 0:
            self.logger.warning("數據清洗後沒有剩餘樣本，無法進行特徵篩選。")
            return []

        # 2. 訓練模型
        self.logger.info(f"開始使用 LightGBM 訓練模型... 樣本數: {len(X)}")
        model = lgb.LGBMClassifier(**self.config.LGBM_PARAMS)
        model.fit(X, y)

        # 3. 提取特徵重要性
        self.logger.info("提取並排序特徵重要性...")
        feature_importances = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.logger.info("\n--- 特徵重要性排名 (前 30) ---\n" + feature_importances.head(30).to_string())

        # 4. 選出最重要的 N 個特徵
        top_features = feature_importances.head(self.config.TOP_N_FEATURES)['feature'].tolist()
        self.logger.info(f"\n--- 已選出最重要的 {self.config.TOP_N_FEATURES} 個特徵 ---")
        for f in top_features:
            self.logger.info(f"- {f}")
            
        return top_features

    def save_selected_features(self, features: List[str]) -> None:
        """將選出的特徵列表儲存為 JSON 檔案。"""
        output_path = self.config.OUTPUT_BASE_DIR / self.config.OUTPUT_FILENAME
        self.logger.info(f"正在將選出的特徵儲存到: {output_path}")
        
        # 為了讓 JSON 檔案更具可讀性，我們儲存一個字典
        output_data = {
            "description": "由 03_feature_selection.py 產生的最佳特徵列表",
            "feature_count": len(features),
            "selected_features": features
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        
        self.logger.info("特徵列表儲存成功。")

    def run(self) -> None:
        """執行完整特徵篩選流程。"""
        self.logger.info("========= 特徵篩選流程開始 =========")
        input_files = self.find_input_files()
        
        if not input_files:
            self.logger.warning("在輸入目錄中沒有找到任何檔案，流程結束。")
            return

        # 為了簡化，我們先處理第一個找到的檔案。
        # 在實際應用中，您可能需要合併多個檔案或循環處理它們。
        target_file = input_files[0]
        self.logger.info(f"--- 正在處理檔案: {target_file.name} ---")
        
        try:
            df = pd.read_parquet(target_file)
            
            # 步驟 1: 創建預測目標
            df_with_target = self.create_labels(df)
            
            # 步驟 2: 進行特徵篩選
            selected_features = self.select_features(df_with_target)
            
            if not selected_features:
                self.logger.error("未能選出任何特徵，流程終止。")
                return

            # 步驟 3: 儲存結果
            self.save_selected_features(selected_features)
            
            self.logger.info(f"--- 檔案 {target_file.name} 處理完畢 ---")

        except Exception as e:
            self.logger.error(f"處理檔案 {target_file.name} 時發生錯誤: {e}", exc_info=True)
            
        self.logger.info("========= 特徵篩選流程結束 =========")

if __name__ == "__main__":
    try:
        config = Config()
        selector = FeatureSelector(config)
        selector.run()
    except Exception as e:
        logging.critical(f"特徵篩選腳本執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
        sys.exit(1)
