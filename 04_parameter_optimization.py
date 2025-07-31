# 檔名: 04_parameter_optimization.py
# 描述: 使用 Optuna 和 Backtrader 為篩選出的特徵尋找最佳參數。
# 版本: 1.7 (策略迭代：使用布林帶反轉策略進行可行性驗證)

import logging
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import backtrader as bt
import optuna
from finta import TA

# ==============================================================================
# 1. 配置區塊
# ==============================================================================
class Config:
    """儲存腳本所需的所有配置參數。"""
    # --- 輸入檔案 ---
    INPUT_FEATURES_FILE = Path("Output_ML_Pipeline/selected_features.json")
    INPUT_DATA_FILE = Path("Output_Data_Pipeline_v2/MarketData/EURUSD_sml/EURUSD_sml_D1.parquet")
    
    # --- 輸出檔案 ---
    OUTPUT_BASE_DIR = Path("Output_ML_Pipeline")
    OUTPUT_FILENAME = "optimal_parameters.json"

    # --- Optuna 優化設定 ---
    N_TRIALS = 150 # 可以適當增加嘗試次數
    OPTIMIZE_METRIC = 'sharpe'
    
    # --- 回測設定 ---
    INITIAL_CASH = 100000.0
    COMMISSION = 0.001
    
    # --- 參數搜索範圍 ---
    PARAMETER_RANGES = {
        'SMA': (5, 200),
        'EMA': (5, 200),
        'RSI': (5, 50),
        'CCI': (5, 50),
        'ATR': (5, 50),
        'MFI': (5, 50),
        'WILLIAMS': (5, 50),
        'MACD_fast': (5, 30),
        'MACD_slow': (20, 80),
        'MACD_signal': (5, 30),
        'DMI_period': (5, 50),
        'BBANDS_period': (10, 100), # 擴大布林帶週期搜索範圍
        'BBANDS_devfactor': (1.5, 4.0), # 擴大標準差倍數搜索範圍
        'STOCH_k': (5, 20),
        'STOCH_d': (2, 10),
    }

    LOG_LEVEL = "INFO"

# ==============================================================================
# 2. Backtrader 策略 (用於 Optuna) - 已升級為布林帶策略
# ==============================================================================
class OptunaStrategy(bt.Strategy):
    """
    *** NEW (v1.7): 使用布林帶反轉策略作為新的可行性驗證基準 ***
    """
    params = (
        ('bb_period', 20),
        ('bb_devfactor', 2.0),
    )

    def __init__(self):
        self.bbands = bt.indicators.BollingerBands(
            self.data.close, 
            period=self.p.bb_period, 
            devfactor=self.p.bb_devfactor
        )

    def next(self):
        if not self.position:  # 如果沒有持倉
            # 當價格觸及或跌破下軌時，買入
            if self.data.close[0] <= self.bbands.lines.bot[0]:
                self.buy()
        else:  # 如果有持倉
            # 當價格觸及或超過上軌時，賣出
            if self.data.close[0] >= self.bbands.lines.top[0]:
                self.sell()

# ==============================================================================
# 3. 參數優化器類別
# ==============================================================================
class ParameterOptimizer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.df_data = self._load_data()
        self.config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

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

    def _load_data(self) -> pd.DataFrame:
        if not self.config.INPUT_DATA_FILE.exists():
            self.logger.critical(f"數據檔案 {self.config.INPUT_DATA_FILE} 不存在！")
            sys.exit(1)
        self.logger.info(f"正在從 {self.config.INPUT_DATA_FILE} 載入回測數據...")
        df = pd.read_parquet(self.config.INPUT_DATA_FILE)
        df.index = pd.to_datetime(df.index)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df

    def objective(self, trial: optuna.trial.Trial) -> float:
        
        # *** NEW (v1.7): 優化目標變為布林帶的參數 ***
        p_period_min, p_period_max = self.config.PARAMETER_RANGES['BBANDS_period']
        p_dev_min, p_dev_max = self.config.PARAMETER_RANGES['BBANDS_devfactor']
        
        bb_period = trial.suggest_int("BBANDS_period", p_period_min, p_period_max)
        bb_devfactor = trial.suggest_float("BBANDS_devfactor", p_dev_min, p_dev_max, step=0.1)

        # --- 執行回測 ---
        cerebro = bt.Cerebro(stdstats=False)
        
        cerebro.addstrategy(OptunaStrategy, 
                            bb_period=bb_period,
                            bb_devfactor=bb_devfactor)
        
        data_feed = bt.feeds.PandasData(dataname=self.df_data)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(self.config.INITIAL_CASH)
        cerebro.broker.setcommission(commission=self.config.COMMISSION)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        try:
            results = cerebro.run()
            strat = results[0]
        except Exception:
            return -100.0

        # --- 提取結果 ---
        trades = strat.analyzers.trades.get_analysis()
        if trades.total.total == 0:
            return -100.0
            
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
        return sharpe if sharpe is not None else -100.0

    def run(self) -> None:
        self.logger.info(f"========= 參數優化流程開始 (v1.7 - 布林帶反轉策略) =========")
        self.logger.info(f"優化目標: {self.config.OPTIMIZE_METRIC.upper()}")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.config.N_TRIALS, show_progress_bar=True)

        self.logger.info("優化完成！")
        if study.best_value <= 0:
            self.logger.warning("警告：未能找到任何能產生正夏普比率的參數組合。")
            self.logger.warning("這可能意味著布林帶反轉策略在該市場上仍然無效，您可能需要考慮更換市場數據或重新進行特徵篩選。")
        else:
            self.logger.info(f"*** 成功！找到可行參數組合，夏普比率為正。可以進入下一階段！ ***")

        self.logger.info(f"最佳 Trial: {study.best_trial.number}")
        self.logger.info(f"最佳分數 ({self.config.OPTIMIZE_METRIC}): {study.best_value:.4f}")
        self.logger.info("最佳參數:")
        for key, value in study.best_params.items():
            self.logger.info(f"  - {key}: {value}")

        # 儲存最佳參數
        output_path = self.config.OUTPUT_BASE_DIR / self.config.OUTPUT_FILENAME
        output_data = {
            "description": f"由 04_parameter_optimization.py (v1.7) 產生的最佳參數 (驗證策略: 布林帶反轉)",
            "best_value": study.best_value,
            "optimal_parameters": study.best_params
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        self.logger.info(f"最佳參數已儲存到: {output_path}")
        self.logger.info("========= 參數優化流程結束 =========")

if __name__ == "__main__":
    try:
        config = Config()
        optimizer = ParameterOptimizer(config)
        optimizer.run()
    except Exception as e:
        logging.critical(f"參數優化腳本執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
        sys.exit(1)
