# 檔名: 04_parameter_optimization.py
# 描述: 使用 Optuna 和 Backtrader 為篩選出的特徵尋找最佳參數。
# 版本: 1.6 (對應迭代式開發架構 v2.0，修正指標實例化錯誤)

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
    N_TRIALS = 100
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
        'BBANDS_period': (10, 50),
        'BBANDS_devfactor': (1.5, 3.5),
        'STOCH_k': (5, 20),
        'STOCH_d': (2, 10),
    }

    LOG_LEVEL = "INFO"

# ==============================================================================
# 2. 自定義指標 (MFI)
# ==============================================================================
class MFIIndicator(bt.Indicator):
    """
    自定義的資金流量指標 (Money Flow Index) - 已強化
    """
    lines = ('mfi',)
    params = (('period', 14),)

    def __init__(self):
        tp = (self.data.close + self.data.high + self.data.low) / 3.0
        mf = tp * self.data.volume
        mf_positive = bt.If(tp > tp(-1), mf, 0)
        mf_negative = bt.If(tp < tp(-1), mf, 0)
        mf_pos_sum = bt.indicators.SumN(mf_positive, period=self.p.period)
        mf_neg_sum = bt.indicators.SumN(mf_negative, period=self.p.period)
        mr = mf_pos_sum / (mf_neg_sum + 1e-9)
        self.lines.mfi = 100.0 - (100.0 / (1.0 + mr))

# ==============================================================================
# 3. Backtrader 策略 (用於 Optuna) - 已升級
# ==============================================================================
class OptunaStrategy(bt.Strategy):
    """
    這個策略接收指標的「類別」和「參數」，並在內部安全地建立實例。
    它使用兩條同類型但不同週期的指標（一快一慢），
    在它們交叉時進行買賣，這是一個穩健的參數評估基準。
    """
    params = (
        ('indicator_class', None),
        ('fast_params', None),
        ('slow_params', None),
    )

    def __init__(self):
        if not self.p.indicator_class or not self.p.fast_params or not self.p.slow_params:
            raise ValueError("必須提供指標類別和快、慢線參數！")
        
        # 在策略內部安全地實例化指標
        fast_indicator = self.p.indicator_class(self.data, **self.p.fast_params)
        slow_indicator = self.p.indicator_class(self.data, **self.p.slow_params)
        
        self.crossover = bt.indicators.CrossOver(fast_indicator, slow_indicator)

    def next(self):
        if self.crossover > 0:  # 快線上穿慢線
            if not self.position:
                self.buy()
        elif self.crossover < 0:  # 快線下穿慢線
            if self.position:
                self.sell()

# ==============================================================================
# 4. 參數優化器類別
# ==============================================================================
class ParameterOptimizer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.selected_features = self._load_selected_features()
        self.indicator_blueprints = self._parse_feature_blueprints()
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

    def _load_selected_features(self) -> List[str]:
        if not self.config.INPUT_FEATURES_FILE.exists():
            self.logger.critical(f"輸入檔案 {self.config.INPUT_FEATURES_FILE} 不存在！")
            sys.exit(1)
        with open(self.config.INPUT_FEATURES_FILE, 'r') as f:
            data = json.load(f)
        self.logger.info(f"成功從 {self.config.INPUT_FEATURES_FILE} 載入 {len(data['selected_features'])} 個特徵。")
        return data['selected_features']

    def _load_data(self) -> pd.DataFrame:
        if not self.config.INPUT_DATA_FILE.exists():
            self.logger.critical(f"數據檔案 {self.config.INPUT_DATA_FILE} 不存在！")
            sys.exit(1)
        self.logger.info(f"正在從 {self.config.INPUT_DATA_FILE} 載入回測數據...")
        df = pd.read_parquet(self.config.INPUT_DATA_FILE)
        df.index = pd.to_datetime(df.index)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df

    def _parse_feature_blueprints(self) -> Dict:
        blueprints = {}
        unique_indicator_types = set()
        for feature in self.selected_features:
            ind_type_match = re.match(r"([A-Z]+)", feature)
            if ind_type_match:
                ind_type = ind_type_match.group(1)
                # 只選擇適合做雙線交叉的指標
                if ind_type in ['SMA', 'EMA']:
                    unique_indicator_types.add(ind_type)

        self.logger.info("已解析出以下適合進行交叉策略優化的指標類型:")
        for ind_type in sorted(list(unique_indicator_types)):
            bt_class = self._get_bt_indicator_class(ind_type)
            if bt_class:
                blueprints[ind_type] = bt_class
                self.logger.info(f"- {ind_type}")
        return blueprints

    def _get_bt_indicator_class(self, name: str):
        mapping = {
            'SMA': bt.indicators.SimpleMovingAverage, 
            'EMA': bt.indicators.ExponentialMovingAverage,
        }
        return mapping.get(name.upper())

    def objective(self, trial: optuna.trial.Trial) -> float:
        if not self.indicator_blueprints:
             return -200.0

        ind_type_to_optimize = trial.suggest_categorical("indicator_type", list(self.indicator_blueprints.keys()))
        ind_class = self.indicator_blueprints[ind_type_to_optimize]
        
        p_range_min, p_range_max = self.config.PARAMETER_RANGES[ind_type_to_optimize]
        
        p_fast_name = f"{ind_type_to_optimize}_fast_period"
        p_fast_val = trial.suggest_int(p_fast_name, p_range_min, p_range_max - 1)
        
        p_slow_name = f"{ind_type_to_optimize}_slow_period"
        p_slow_val = trial.suggest_int(p_slow_name, p_fast_val + 1, p_range_max)

        # --- 執行回測 ---
        cerebro = bt.Cerebro(stdstats=False)
        
        fast_params = {'period': p_fast_val}
        slow_params = {'period': p_slow_val}
        
        cerebro.addstrategy(OptunaStrategy, 
                            indicator_class=ind_class,
                            fast_params=fast_params, 
                            slow_params=slow_params)
        
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
        if not self.indicator_blueprints:
            self.logger.error("沒有解析出任何適合進行交叉策略優化的指標 (如 SMA, EMA)。")
            return

        self.logger.info(f"========= 參數優化流程開始 (v1.6 - 雙均線交叉) =========")
        self.logger.info(f"優化目標: {self.config.OPTIMIZE_METRIC.upper()}")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.config.N_TRIALS, show_progress_bar=True)

        self.logger.info("優化完成！")
        if study.best_value <= 0:
            self.logger.warning("警告：未能找到任何能產生正夏普比率的參數組合。")
            self.logger.warning("這可能意味著雙均線交叉策略在該市場上無效，或者需要調整參數範圍。")
        
        self.logger.info(f"最佳 Trial: {study.best_trial.number}")
        self.logger.info(f"最佳分數 ({self.config.OPTIMIZE_METRIC}): {study.best_value:.4f}")
        self.logger.info("最佳參數:")
        best_indicator = study.best_params.get("indicator_type")
        self.logger.info(f"  - 最佳指標類型: {best_indicator}")
        for key, value in study.best_params.items():
            if key != "indicator_type":
                 self.logger.info(f"    - {key}: {value}")

        # 儲存最佳參數
        output_path = self.config.OUTPUT_BASE_DIR / self.config.OUTPUT_FILENAME
        output_data = {
            "description": f"由 04_parameter_optimization.py (v1.6) 產生的最佳參數 (優化目標: {self.config.OPTIMIZE_METRIC})",
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
