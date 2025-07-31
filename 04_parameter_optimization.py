# 檔名: 04_ml_model_optimization.py
# 描述: 整合 ML 模型超參數優化與最終樣本外回測的完整流程。
# 版本: 3.3 (修正因數據中存在 inf 值導致的繪圖錯誤)

import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import backtrader as bt
import lightgbm as lgb
from finta import TA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import optuna

# ==============================================================================
# 1. 配置區塊
# ==============================================================================
class Config:
    """儲存腳本所需的所有配置參數。"""
    # --- 輸入檔案 ---
    INPUT_FEATURES_FILE = Path("Output_ML_Pipeline/selected_features.json")
    INPUT_DATA_DIR = Path("Output_Feature_Engineering/MarketData_with_All_Features")
    
    # --- 輸出設定 ---
    OUTPUT_BASE_DIR = Path("Output_ML_Pipeline")
    
    # --- 數據分割 ---
    TRAIN_VALIDATION_SPLIT_DATE = "2023-01-01" # 用於分割訓練集和驗證集
    OUT_OF_SAMPLE_START_DATE = "2024-01-01"    # 樣本外測試的起始日期

    # --- Labeling (與 03 保持一致) ---
    LABEL_LOOK_FORWARD_PERIODS: int = 5
    LABEL_RETURN_THRESHOLD: float = 0.005

    # --- Optuna 優化設定 ---
    N_TRIALS = 100 # 優化嘗試次數
    
    # --- 回測設定 ---
    INITIAL_CASH = 100000.0
    COMMISSION = 0.001
    ENTRY_PROB_THRESHOLD = 0.55

    LOG_LEVEL = "INFO"

# ==============================================================================
# 2. Backtrader 最終策略
# ==============================================================================
class FinalMLStrategy(bt.Strategy):
    params = (
        ('model', None),
        ('features', None),
        ('entry_threshold', 0.55),
    )

    def __init__(self):
        if not self.p.model or not self.p.features:
            raise ValueError("模型和特徵列表必須提供！")
        
        self.feature_lines = []
        for feature_name in self.p.features:
            self.feature_lines.append(getattr(self.data.lines, feature_name))

    def next(self):
        feature_vector = []
        try:
            for line in self.feature_lines:
                feature_vector.append(line[0])
        except IndexError:
            return

        feature_vector = np.array(feature_vector).reshape(1, -1)
        pred_prob = self.p.model.predict_proba(feature_vector)[0][1]

        if not self.position:
            if pred_prob > self.p.entry_threshold:
                self.buy()
        else:
            if pred_prob < 0.5:
                self.sell()

# ==============================================================================
# 3. 主流程控制器
# ==============================================================================
class MLOptimizerAndBacktester:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        
        self.selected_features = self._load_json(self.config.INPUT_FEATURES_FILE)['selected_features']
        self.full_df = self._load_and_prepare_data()
        
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.LOG_LEVEL.upper())
        if logger.hasHandlers():
            logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return logger

    def _load_json(self, file_path: Path) -> Dict:
        if not file_path.exists():
            self.logger.critical(f"輸入檔案 {file_path} 不存在！請先執行 03_feature_selection.py。")
            sys.exit(1)
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.logger.info(f"成功從 {file_path} 載入特徵列表。")
        return data

    def _load_and_prepare_data(self) -> pd.DataFrame:
        """載入所有數據，合併，並創建 Label。"""
        input_files = list(self.config.INPUT_DATA_DIR.rglob("*.parquet"))
        if not input_files:
            self.logger.critical(f"在 {self.config.INPUT_DATA_DIR} 中找不到任何數據檔案！")
            sys.exit(1)
        
        all_dfs = [pd.read_parquet(f) for f in input_files]
        df = pd.concat(all_dfs).sort_index()
        self.logger.info(f"已合併 {len(input_files)} 個檔案，共 {len(df)} 筆數據。")

        future_returns = df['close'].shift(-self.config.LABEL_LOOK_FORWARD_PERIODS) / df['close'] - 1
        df['target'] = (future_returns > self.config.LABEL_RETURN_THRESHOLD).astype(int)
        
        missing_features = [f for f in self.selected_features if f not in df.columns]
        if missing_features:
            self.logger.warning(f"數據中缺少以下特徵，將被忽略: {missing_features}")
            self.selected_features = [f for f in self.selected_features if f in df.columns]

        # *** NEW (v3.3): 清理無窮大值，以防止繪圖錯誤 ***
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.logger.info("已將數據中的無窮大值替換為 NaN。")

        df.dropna(inplace=True)
        return df

    def objective(self, trial: optuna.trial.Trial) -> float:
        """Optuna 的目標函數，用於尋找最佳超參數。"""
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        }

        model = lgb.LGBMClassifier(**param)
        model.fit(self.X_train.values, self.y_train.values)
        
        preds = model.predict(self.X_val.values)
        accuracy = accuracy_score(self.y_val.values, preds)
        return accuracy

    def run(self):
        """執行完整的優化與回測流程。"""
        self.logger.info("========= 整合式優化與回測流程開始 (v3.3) =========")

        # 1. 數據分割
        df_in_sample = self.full_df[self.full_df.index < self.config.OUT_OF_SAMPLE_START_DATE]
        df_out_of_sample = self.full_df[self.full_df.index >= self.config.OUT_OF_SAMPLE_START_DATE]

        df_train = df_in_sample[df_in_sample.index < self.config.TRAIN_VALIDATION_SPLIT_DATE]
        df_val = df_in_sample[df_in_sample.index >= self.config.TRAIN_VALIDATION_SPLIT_DATE]

        self.X_train = df_train[self.selected_features]
        self.y_train = df_train['target']
        self.X_val = df_val[self.selected_features]
        self.y_val = df_val['target']
        
        self.logger.info(f"數據分割完成: 訓練集({len(df_train)}), 驗證集({len(df_val)}), 樣本外測試集({len(df_out_of_sample)})")

        # 2. 執行 Optuna 超參數優化
        self.logger.info("--- 開始執行 LightGBM 超參數優化 ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.config.N_TRIALS, show_progress_bar=True)
        
        self.logger.info("超參數優化完成！")
        self.logger.info(f"最佳驗證集準確率: {study.best_value:.4f}")
        self.logger.info("最佳超參數:")
        for key, value in study.best_params.items():
            self.logger.info(f"  - {key}: {value}")

        # 3. 訓練最終模型
        self.logger.info("--- 使用最佳超參數訓練最終模型 ---")
        X_in_sample = df_in_sample[self.selected_features]
        y_in_sample = df_in_sample['target']
        
        final_model = lgb.LGBMClassifier(**study.best_params)
        final_model.fit(X_in_sample.values, y_in_sample.values)
        self.logger.info("最終模型訓練完成。")

        # 4. 執行樣本外最終回測
        self.logger.info("--- 開始執行樣本外最終回測 ---")
        self.run_final_backtest(df_out_of_sample, final_model)

        self.logger.info("========= 流程結束 =========")

    def run_final_backtest(self, df: pd.DataFrame, model: lgb.LGBMClassifier):
        """執行單次回測並打印報告。"""
        class PandasDataWithFeatures(bt.feeds.PandasData):
            lines = tuple(self.selected_features)
            params = tuple([(f, -1) for f in self.selected_features])

        data_feed = PandasDataWithFeatures(dataname=df)
        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed)
        cerebro.addstrategy(FinalMLStrategy, model=model, features=self.selected_features,
                            entry_threshold=self.config.ENTRY_PROB_THRESHOLD)
        
        cerebro.broker.setcash(self.config.INITIAL_CASH)
        cerebro.broker.setcommission(commission=self.config.COMMISSION)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        results = cerebro.run()
        
        # 打印報告
        strat = results[0]
        analysis = strat.analyzers
        trades = analysis.trades.get_analysis()
        
        self.logger.info("\n" + "="*50)
        self.logger.info("樣本外最終回測績效報告")
        self.logger.info("="*50)
        self.logger.info(f"最終資產價值: {cerebro.broker.getvalue():,.2f}")
        self.logger.info(f"總淨利: {trades.pnl.net.total:,.2f}")
        
        sharpe_ratio = analysis.sharpe.get_analysis().get('sharperatio')
        if sharpe_ratio is not None:
            self.logger.info(f"夏普比率: {sharpe_ratio:.2f}")
        else:
            self.logger.info("夏普比率: N/A (無交易或無波動)")
            
        self.logger.info(f"最大回撤: {analysis.drawdown.get_analysis().max.drawdown:.2f}%")
        self.logger.info(f"總交易次數: {trades.total.total}")
        if trades.total.total > 0:
            self.logger.info(f"勝率: {trades.won.total / trades.total.total:.2%}")
        self.logger.info("="*50)

        # 繪製圖表
        plot_path = self.config.OUTPUT_BASE_DIR / "final_backtest_chart.png"
        fig = cerebro.plot(style='candlestick', iplot=False)[0][0]
        fig.savefig(plot_path, dpi=300)
        self.logger.info(f"最終回測圖表已儲存至: {plot_path}")

if __name__ == "__main__":
    try:
        config = Config()
        optimizer = MLOptimizerAndBacktester(config)
        optimizer.run()
    except Exception as e:
        logging.critical(f"腳本執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
        sys.exit(1)
