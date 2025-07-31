# 檔名: 04_parameter_optimization.py
# 描述: 使用 Optuna 和 Backtrader 為篩選出的特徵尋找最佳參數。
# 版本: 1.8 (升級為可配置的優化引擎，支持不同超參數調校演算法)

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
    INPUT_DATA_FILE = Path("Output_Data_Pipeline_v2/MarketData/EURUSD_sml/EURUSD_sml_D1.parquet")
    
    # --- 輸出檔案 ---
    OUTPUT_BASE_DIR = Path("Output_ML_Pipeline")
    OUTPUT_FILENAME = "optimal_parameters.json"

    # --- Optuna 優化設定 ---
    N_TRIALS = 150
    OPTIMIZE_METRIC = 'sharpe'
    
    # *** NEW (v1.8): 選擇您的超參數調校演算法 ***
    # 'TPE': 貝葉斯優化 (預設，高效)
    # 'Grid': 網格搜索 (窮舉，最慢)
    # 'Random': 隨機搜索
    SAMPLER = 'TPE' 
    
    # 是否啟用剪枝 (Pruning): 自動提早結束沒有潛力的試驗，以節省時間
    PRUNING = True

    # --- 回測設定 ---
    INITIAL_CASH = 100000.0
    COMMISSION = 0.001
    
    # --- 參數搜索範圍 ---
    # 對於 'Grid' 搜索，列表中的每個值都會被測試
    # 對於 'TPE' 和 'Random' 搜索，會在元組 (min, max) 範圍內尋找
    PARAMETER_RANGES = {
        'BBANDS_period': [10, 20, 30, 40, 50, 60], # Grid Search 的範例
        'BBANDS_devfactor': [1.8, 2.0, 2.2, 2.5, 3.0], # Grid Search 的範例
    }
    # TPE/Random 的搜索範圍
    PARAMETER_RANGES_TPE = {
        'BBANDS_period': (10, 100),
        'BBANDS_devfactor': (1.5, 4.0),
    }

    LOG_LEVEL = "INFO"

# ==============================================================================
# 2. Backtrader 策略 (用於 Optuna) - 布林帶策略
# ==============================================================================
class OptunaStrategy(bt.Strategy):
    """
    使用布林帶反轉策略作為可行性驗證基準
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
        if not self.position:
            if self.data.close[0] <= self.bbands.lines.bot[0]:
                self.buy()
        else:
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
        
        if self.config.SAMPLER == 'Grid':
            # Grid Search 從預定義的列表中選擇
            bb_period = trial.suggest_categorical("BBANDS_period", self.config.PARAMETER_RANGES['BBANDS_period'])
            bb_devfactor = trial.suggest_categorical("BBANDS_devfactor", self.config.PARAMETER_RANGES['BBANDS_devfactor'])
        else:
            # TPE/Random 從範圍中選擇
            p_period_min, p_period_max = self.config.PARAMETER_RANGES_TPE['BBANDS_period']
            p_dev_min, p_dev_max = self.config.PARAMETER_RANGES_TPE['BBANDS_devfactor']
            bb_period = trial.suggest_int("BBANDS_period", p_period_min, p_period_max)
            bb_devfactor = trial.suggest_float("BBANDS_devfactor", p_dev_min, p_dev_max, step=0.1)

        # --- 執行回測 ---
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.addstrategy(OptunaStrategy, bb_period=bb_period, bb_devfactor=bb_devfactor)
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
        
        # 向 Pruner 報告結果
        trial.report(sharpe, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return sharpe if sharpe is not None else -100.0

    def run(self) -> None:
        self.logger.info(f"========= 參數優化流程開始 (v1.8 - 優化引擎) =========")
        self.logger.info(f"優化演算法: {self.config.SAMPLER.upper()}")
        
        # *** NEW (v1.8): 根據設定選擇 Sampler ***
        sampler = None
        if self.config.SAMPLER.lower() == 'grid':
            search_space = {
                "BBANDS_period": self.config.PARAMETER_RANGES['BBANDS_period'],
                "BBANDS_devfactor": self.config.PARAMETER_RANGES['BBANDS_devfactor']
            }
            sampler = optuna.samplers.GridSampler(search_space)
            # Grid Search 會測試所有組合，N_TRIALS 應設為組合總數
            self.config.N_TRIALS = len(search_space['BBANDS_period']) * len(search_space['BBANDS_devfactor'])
            self.logger.info(f"網格搜索已啟用，將執行全部 {self.config.N_TRIALS} 種組合。")
        elif self.config.SAMPLER.lower() == 'random':
            sampler = optuna.samplers.RandomSampler()
        else: # 預設為 TPE (貝葉斯優化)
            sampler = optuna.samplers.TPESampler()

        # 設定 Pruner
        pruner = optuna.pruners.MedianPruner() if self.config.PRUNING else optuna.pruners.NopPruner()

        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(self.objective, n_trials=self.config.N_TRIALS, show_progress_bar=True)

        self.logger.info("優化完成！")
        if not study.best_trial or study.best_value <= 0:
            self.logger.warning("警告：未能找到任何能產生正夏普比率的參數組合。")
            self.logger.warning("這可能意味著當前的驗證策略在該市場上無效，您可能需要考慮更換策略或市場數據。")
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
                "description": f"由 04_parameter_optimization.py (v1.8) 產生的最佳參數 (驗證策略: 布林帶反轉)",
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
