# 檔名: 04_parameter_optimization.py
# 描述: 使用 Optuna 和 Backtrader 為篩選出的特徵尋找最佳參數。
# 版本: 1.3 (簡化策略邏輯以專注於參數評估，並強化回饋機制)

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
        'SMA': (5, 200), # 擴大範圍以進行更廣泛的搜索
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
        # *** FIX: 增加一個極小的數值 (epsilon) 來避免除以零的錯誤 ***
        mr = mf_pos_sum / (mf_neg_sum + 1e-9)
        self.lines.mfi = 100.0 - (100.0 / (1.0 + mr))

# ==============================================================================
# 3. Backtrader 策略 (用於 Optuna) - 已簡化
# ==============================================================================
class OptunaStrategy(bt.Strategy):
    """
    *** NEW: 一個極其簡化的策略，專門用於評估單一指標參數的表現 ***
    這個策略只使用一個指標（例如，一個移動平均線），並執行簡單的交叉買賣。
    目的是為了確保能產生交易，從而有效地測試不同參數設定的優劣。
    """
    params = (
        ('indicator_class', None),
        ('indicator_params', None),
    )

    def __init__(self):
        if not self.p.indicator_class or self.p.indicator_params is None:
            raise ValueError("必須提供指標類別和參數！")
        
        # 創建本次 trial 需要評估的單一指標
        self.the_indicator = self.p.indicator_class(self.data, **self.p.indicator_params)
        
        # 創建一個價格與指標的交叉信號
        self.crossover = bt.indicators.CrossOver(self.data.close, self.the_indicator)

    def next(self):
        if self.crossover > 0:  # 價格上穿指標
            if not self.position:
                self.buy()
        elif self.crossover < 0:  # 價格下穿指標
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
            if feature in ['MACD', 'SIGNAL']:
                unique_indicator_types.add('MACD')
                continue
            if feature in ['DI+', 'DI-']:
                unique_indicator_types.add('DMI')
                continue
            if 'BB_' in feature:
                unique_indicator_types.add('BBANDS')
                continue
            if 'STOCH' in feature:
                unique_indicator_types.add('STOCH')
                continue
            match = re.match(r"([A-Z]+)", feature)
            if match:
                unique_indicator_types.add(match.group(1))

        self.logger.info("已解析出以下需要優化的指標類型:")
        for ind_type in sorted(list(unique_indicator_types)):
            if ind_type in self.config.PARAMETER_RANGES or any(key.startswith(ind_type) for key in self.config.PARAMETER_RANGES):
                bt_class = self._get_bt_indicator_class(ind_type)
                if bt_class:
                    blueprints[ind_type] = bt_class
                    self.logger.info(f"- {ind_type}")
        return blueprints

    def _get_bt_indicator_class(self, name: str):
        mapping = {
            'SMA': bt.indicators.SimpleMovingAverage, 'EMA': bt.indicators.ExponentialMovingAverage,
            'RSI': bt.indicators.RSI, 'MACD': bt.indicators.MACD, 'CCI': bt.indicators.CCI,
            'ATR': bt.indicators.ATR, 'BBANDS': bt.indicators.BollingerBands,
            'STOCH': bt.indicators.Stochastic, 'WILLIAMS': bt.indicators.WilliamsR,
            'MFI': MFIIndicator, 'DMI': bt.indicators.DMI,
        }
        return mapping.get(name.upper())

    def objective(self, trial: optuna.trial.Trial) -> float:
        # *** NEW: 在每個 trial 中，只選擇一個指標類型進行測試 ***
        # 這能讓我們獨立地評估每個指標的最佳參數，避免不同指標間的相互干擾。
        ind_type_to_optimize = trial.suggest_categorical("indicator_type", list(self.indicator_blueprints.keys()))
        ind_class = self.indicator_blueprints[ind_type_to_optimize]
        
        current_params = {}
        # 根據選擇的指標類型，生成對應的參數
        if ind_type_to_optimize in ['SMA', 'EMA', 'RSI', 'CCI', 'ATR', 'MFI', 'WILLIAMS', 'DMI']:
            p_name = f"{ind_type_to_optimize}_period"
            min_v, max_v = self.config.PARAMETER_RANGES[ind_type_to_optimize]
            current_params['period'] = trial.suggest_int(p_name, min_v, max_v)
        elif ind_type_to_optimize == 'MACD':
            p1 = trial.suggest_int('MACD_fast', *self.config.PARAMETER_RANGES['MACD_fast'])
            p2 = trial.suggest_int('MACD_slow', *self.config.PARAMETER_RANGES['MACD_slow'])
            p3 = trial.suggest_int('MACD_signal', *self.config.PARAMETER_RANGES['MACD_signal'])
            if p2 <= p1: p2 = p1 + 1
            current_params = {'period_me1': p1, 'period_me2': p2, 'period_signal': p3}
        elif ind_type_to_optimize == 'BBANDS':
             period = trial.suggest_int('BBANDS_period', *self.config.PARAMETER_RANGES['BBANDS_period'])
             dev = trial.suggest_float('BBANDS_devfactor', *self.config.PARAMETER_RANGES['BBANDS_devfactor'], step=0.1)
             current_params = {'period': period, 'devfactor': dev}
        
        if not current_params: return -100.0

        # --- 執行回測 ---
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.addstrategy(OptunaStrategy, indicator_class=ind_class, indicator_params=current_params)
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
        # *** FIX: 強化回饋機制 ***
        trades = strat.analyzers.trades.get_analysis()
        if trades.total.total == 0:
            return -100.0  # 如果沒有交易，給予一個很大的負分
            
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
        return sharpe if sharpe is not None else -100.0

    def run(self) -> None:
        if not self.indicator_blueprints:
            self.logger.error("沒有解析出任何可優化的指標。")
            return

        self.logger.info(f"========= 參數優化流程開始 (v1.3) =========")
        self.logger.info(f"優化目標: {self.config.OPTIMIZE_METRIC.upper()}")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.config.N_TRIALS, show_progress_bar=True)

        self.logger.info("優化完成！")
        if study.best_value <= 0:
            self.logger.warning("警告：未能找到任何能產生正夏普比率的參數組合。")
            self.logger.warning("這可能意味著單一指標交叉策略在該市場上無效，或者需要調整參數範圍。")
        
        self.logger.info(f"最佳 Trial: {study.best_trial.number}")
        self.logger.info(f"最佳分數 ({self.config.OPTIMIZE_METRIC}): {study.best_value:.4f}")
        self.logger.info("最佳參數:")
        # *** NEW: 整理輸出，只顯示與最佳指標相關的參數 ***
        best_indicator = study.best_params.get("indicator_type")
        self.logger.info(f"  - 最佳指標類型: {best_indicator}")
        for key, value in study.best_params.items():
            if key != "indicator_type":
                 self.logger.info(f"    - {key}: {value}")

        # 儲存最佳參數
        output_path = self.config.OUTPUT_BASE_DIR / self.config.OUTPUT_FILENAME
        output_data = {
            "description": f"由 04_parameter_optimization.py 產生的最佳參數 (優化目標: {self.config.OPTIMIZE_METRIC})",
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
