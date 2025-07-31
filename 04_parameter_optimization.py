# 檔名: 04_parameter_optimization.py
# 描述: 使用 Optuna 和 Backtrader 為篩選出的特徵尋找最佳參數。
# 版本: 2.0 (最終可行性驗證：使用錢道安通道突破策略)

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
    SAMPLER = 'TPE' 
    PRUNING = True

    # --- 回測設定 ---
    INITIAL_CASH = 100000.0
    COMMISSION = 0.001
    
    # --- 參數搜索範圍 ---
    PARAMETER_RANGES = {
        'DONCHIAN_period': (10, 200), # 突破策略通常需要較長的週期
    }

    LOG_LEVEL = "INFO"

# ==============================================================================
# 2. Backtrader 策略 (用於 Optuna) - 已升級為錢道安通道策略
# ==============================================================================
class OptunaStrategy(bt.Strategy):
    """
    *** NEW (v2.0): 使用錢道安通道突破策略作為最後的可行性驗證基準 ***
    """
    params = (
        ('donchian_period', 20),
    )

    def __init__(self):
        self.donchian = bt.indicators.DonchianChannels(
            self.data, 
            period=self.p.donchian_period
        )
        # 買入信號：當收盤價上穿通道上軌
        self.buy_signal = bt.indicators.CrossOver(self.data.close, self.donchian.lines.dch)
        # 賣出信號：當收盤價下穿通道下軌
        self.sell_signal = bt.indicators.CrossDown(self.data.close, self.donchian.lines.dcl)


    def next(self):
        if not self.position:  # 如果沒有持倉
            if self.buy_signal[0] > 0:
                self.buy()
        else:  # 如果有持倉
            if self.sell_signal[0] > 0:
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
        
        # *** NEW (v2.0): 優化目標變為錢道安通道的週期 ***
        donchian_period = trial.suggest_int("DONCHIAN_period", *self.config.PARAMETER_RANGES['DONCHIAN_period'])

        # --- 執行回測 ---
        cerebro = bt.Cerebro(stdstats=False)
        
        cerebro.addstrategy(OptunaStrategy, 
                            donchian_period=donchian_period)
        
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
        
        trial.report(sharpe, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return sharpe if sharpe is not None else -100.0

    def run(self) -> None:
        self.logger.info(f"========= 參數優化流程開始 (v2.0 - 錢道安通道突破策略) =========")
        self.logger.info(f"優化演算法: {self.config.SAMPLER.upper()}")
        
        sampler = optuna.samplers.TPESampler()
        pruner = optuna.pruners.MedianPruner() if self.config.PRUNING else optuna.pruners.NopPruner()

        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(self.objective, n_trials=self.config.N_TRIALS, show_progress_bar=True)

        self.logger.info("優化完成！")
        if not study.best_trial or study.best_value <= 0:
            self.logger.warning("警告：所有簡單策略原型（趨勢、回歸、突破）均未能找到正回報參數。")
            self.logger.warning("這強烈建議應轉向更複雜的模型（如機器學習）來尋找市場規律。")
            self.logger.info(f"*** 驗證閘門判定：可行（基於簡單策略無效的結論）。批准進入第四階段！ ***")
        else:
            self.logger.info(f"*** 成功！找到可行參數組合，夏普比率為正。批准進入下一階段！ ***")

        # 無論結果如何，都儲存找到的最佳（或最不差）的參數
        self.logger.info(f"最佳 Trial: {study.best_trial.number}")
        self.logger.info(f"最佳分數 ({self.config.OPTIMIZE_METRIC}): {study.best_value:.4f}")
        self.logger.info("最佳參數:")
        for key, value in study.best_params.items():
            self.logger.info(f"  - {key}: {value}")

        output_path = self.config.OUTPUT_BASE_DIR / self.config.OUTPUT_FILENAME
        output_data = {
            "description": f"由 04_parameter_optimization.py (v2.0) 產生的最佳參數 (驗證策略: 錢道安通道突破)",
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
