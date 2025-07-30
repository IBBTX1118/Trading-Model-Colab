# 檔名: 04_parameter_optimization.py
# 描述: 使用 Optuna 和 Backtrader 為篩選出的特徵尋找最佳參數。
# 版本: 1.2 (新增自定義 MFI 指標，修復 AttributeError)

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
    # 注意：參數優化通常在單一代表性市場上進行，以節省時間。
    # 這裡我們選擇 EURUSD D1 作為基準數據。您可以根據需求更改。
    INPUT_DATA_FILE = Path("Output_Data_Pipeline_v2/MarketData/EURUSD_sml/EURUSD_sml_D1.parquet")
    
    # --- 輸出檔案 ---
    OUTPUT_BASE_DIR = Path("Output_ML_Pipeline")
    OUTPUT_FILENAME = "optimal_parameters.json"

    # --- Optuna 優化設定 ---
    N_TRIALS = 100  # 優化嘗試次數，可根據需求增加
    OPTIMIZE_METRIC = 'sharpe'  # 優化目標: 'sharpe' (夏普比率), 'sqn' (系統品質指標), 'pnl' (淨利)
    
    # --- 回測設定 ---
    INITIAL_CASH = 100000.0
    COMMISSION = 0.001  # 手續費
    
    # --- 參數搜索範圍 ---
    # 定義每個指標類型參數的搜索範圍
    # 格式: 'INDICATOR_TYPE': (min_value, max_value)
    PARAMETER_RANGES = {
        'SMA': (5, 100),
        'EMA': (5, 100),
        'RSI': (5, 50),
        'CCI': (5, 50),
        'ATR': (5, 50),
        'MFI': (5, 50),
        'WILLIAMS': (5, 50),
        # 對於返回多個值的指標，需要分別定義
        'MACD_fast': (5, 20),
        'MACD_slow': (21, 60),
        'MACD_signal': (5, 20),
        'DMI_period': (5, 50), # DI+ 和 DI- 共用同一個週期參數
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
    自定義的資金流量指標 (Money Flow Index)
    """
    lines = ('mfi',)
    params = (('period', 14),)

    def __init__(self):
        # 計算典型價格 (Typical Price)
        tp = (self.data.close + self.data.high + self.data.low) / 3.0
        # 計算資金流量 (Money Flow)
        mf = tp * self.data.volume

        # 比較今天和昨天的典型價格
        mf_positive = bt.If(tp > tp(-1), mf, 0)
        mf_negative = bt.If(tp < tp(-1), mf, 0)
        
        # 計算 N 週期的正負資金流量總和
        mf_pos_sum = bt.indicators.SumN(mf_positive, period=self.p.period)
        mf_neg_sum = bt.indicators.SumN(mf_negative, period=self.p.period)

        # 計算資金比率 (Money Ratio)
        mr = mf_pos_sum / mf_neg_sum
        
        # 計算 MFI
        self.lines.mfi = 100.0 - (100.0 / (1.0 + mr))


# ==============================================================================
# 3. Backtrader 策略 (用於 Optuna)
# ==============================================================================
class OptunaStrategy(bt.Strategy):
    """
    一個通用的 Backtrader 策略，用於在 Optuna 優化過程中評估參數。
    這個策略的邏輯被刻意簡化，以加快回測速度。
    它使用一個主要的指標 (如 EMA) 作為趨勢濾網，
    並使用另一個指標 (如 RSI) 作為進出場觸發器。
    """
    params = (
        ('trial', None),
        ('indicator_params', None),
    )

    def __init__(self):
        self.inds = {}
        # 創建在本次 trial 中需要的所有指標
        for name, p_config in self.p.indicator_params.items():
            bt_indicator_class = p_config['class']
            params_dict = p_config['params']
            self.inds[name] = bt_indicator_class(self.data, **params_dict)

        # 為了簡化，我們假設第一個 EMA/SMA 是趨勢指標
        # 第二個 RSI/CCI 是震盪指標
        self.trend_indicator = None
        self.oscillator = None
        
        # 尋找一個趨勢指標和一個震盪指標
        for name, ind in self.inds.items():
            if 'EMA' in name.upper() or 'SMA' in name.upper():
                if self.trend_indicator is None:
                    self.trend_indicator = ind
            if 'RSI' in name.upper() or 'CCI' in name.upper() or 'WILLIAMS' in name.upper() or 'MFI' in name.upper():
                 if self.oscillator is None:
                    self.oscillator = ind
        
        # 如果沒有找到特定指標，則使用預設
        if self.trend_indicator is None:
            self.trend_indicator = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        if self.oscillator is None:
            self.oscillator = bt.indicators.RSI(self.data.close, period=14)


    def next(self):
        # 簡單的交易邏輯
        if not self.position:  # 沒有持倉
            # 趨勢向上且震盪指標超賣
            if self.data.close[0] > self.trend_indicator[0] and self.oscillator[0] < 30:
                self.buy()
        else:  # 有持倉
            # 震盪指標超買
            if self.oscillator[0] > 70:
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
        # Optuna 本身很吵，關閉它的日誌，除非有錯誤
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return logger

    def _load_selected_features(self) -> List[str]:
        """載入由 03_feature_selection.py 產生的特徵列表。"""
        if not self.config.INPUT_FEATURES_FILE.exists():
            self.logger.critical(f"輸入檔案 {self.config.INPUT_FEATURES_FILE} 不存在！請先執行 03_feature_selection.py。")
            sys.exit(1)
        
        with open(self.config.INPUT_FEATURES_FILE, 'r') as f:
            data = json.load(f)
        self.logger.info(f"成功從 {self.config.INPUT_FEATURES_FILE} 載入 {len(data['selected_features'])} 個特徵。")
        return data['selected_features']

    def _load_data(self) -> pd.DataFrame:
        """載入回測所需的原始市場數據。"""
        if not self.config.INPUT_DATA_FILE.exists():
            self.logger.critical(f"數據檔案 {self.config.INPUT_DATA_FILE} 不存在！")
            sys.exit(1)
        
        self.logger.info(f"正在從 {self.config.INPUT_DATA_FILE} 載入回測數據...")
        df = pd.read_parquet(self.config.INPUT_DATA_FILE)
        df.index = pd.to_datetime(df.index)
        # Backtrader 需要特定的欄位名稱
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df

    def _parse_feature_blueprints(self) -> Dict:
        """從特徵名稱解析出指標類型，為優化做準備。"""
        blueprints = {}
        unique_indicator_types = set()

        for feature in self.selected_features:
            # 處理 'MACD', 'SIGNAL' -> 歸類為 'MACD'
            if feature in ['MACD', 'SIGNAL']:
                unique_indicator_types.add('MACD')
                continue
            # 處理 'DI+', 'DI-' -> 歸類為 'DMI'
            if feature in ['DI+', 'DI-']:
                unique_indicator_types.add('DMI')
                continue
            # 處理 'BB_UPPER', 'BB_LOWER', 'BB_MIDDLE' -> 歸類為 'BBANDS'
            if 'BB_' in feature:
                unique_indicator_types.add('BBANDS')
                continue
            # 處理 '14 period STOCH %K' -> 歸類為 'STOCH'
            if 'STOCH' in feature:
                unique_indicator_types.add('STOCH')
                continue

            # 處理其他常規指標，如 'RSI_14', 'EMA_50'
            match = re.match(r"([A-Z]+)", feature)
            if match:
                unique_indicator_types.add(match.group(1))

        self.logger.info("已解析出以下需要優化的指標類型:")
        for ind_type in sorted(list(unique_indicator_types)):
             # 確保該指標類型在我們的參數範圍定義中
            if ind_type in self.config.PARAMETER_RANGES or f"{ind_type}_period" in self.config.PARAMETER_RANGES or f"{ind_type}_fast" in self.config.PARAMETER_RANGES:
                bt_class = self._get_bt_indicator_class(ind_type)
                if bt_class:
                    blueprints[ind_type] = bt_class
                    self.logger.info(f"- {ind_type}")
        
        return blueprints

    def _get_bt_indicator_class(self, name: str):
        """將指標名稱映射到 Backtrader 的指標類。"""
        mapping = {
            'SMA': bt.indicators.SimpleMovingAverage,
            'EMA': bt.indicators.ExponentialMovingAverage,
            'RSI': bt.indicators.RSI,
            'MACD': bt.indicators.MACD,
            'CCI': bt.indicators.CCI,
            'ATR': bt.indicators.ATR,
            'BBANDS': bt.indicators.BollingerBands,
            'STOCH': bt.indicators.Stochastic,
            'WILLIAMS': bt.indicators.WilliamsR,
            'MFI': MFIIndicator, # *** FIX: 指向我們自定義的 MFI 指標 ***
            'DMI': bt.indicators.DMI, # DI+ 和 DI- 都從 DMI 計算而來
        }
        return mapping.get(name.upper())

    def objective(self, trial: optuna.trial.Trial) -> float:
        """Optuna 的目標函數。"""
        indicator_params_for_strategy = {}
        
        # 根據 blueprint 為本次 trial 生成參數
        for ind_type, ind_class in self.indicator_blueprints.items():
            if not ind_class:
                continue

            current_params = {}
            if ind_type in ['SMA', 'EMA', 'RSI', 'CCI', 'ATR', 'MFI', 'WILLIAMS', 'DMI']:
                p_name = f"{ind_type}_period"
                range_key = ind_type if ind_type in self.config.PARAMETER_RANGES else f"{ind_type}_period"
                min_v, max_v = self.config.PARAMETER_RANGES[range_key]
                current_params['period'] = trial.suggest_int(p_name, min_v, max_v)

            elif ind_type == 'MACD':
                p1 = trial.suggest_int('MACD_fast', *self.config.PARAMETER_RANGES['MACD_fast'])
                p2 = trial.suggest_int('MACD_slow', *self.config.PARAMETER_RANGES['MACD_slow'])
                p3 = trial.suggest_int('MACD_signal', *self.config.PARAMETER_RANGES['MACD_signal'])
                if p2 <= p1: p2 = p1 + 1
                current_params = {'period_me1': p1, 'period_me2': p2, 'period_signal': p3}

            elif ind_type == 'BBANDS':
                 period = trial.suggest_int('BBANDS_period', *self.config.PARAMETER_RANGES['BBANDS_period'])
                 dev = trial.suggest_float('BBANDS_devfactor', *self.config.PARAMETER_RANGES['BBANDS_devfactor'], step=0.1)
                 current_params = {'period': period, 'devfactor': dev}
            
            if current_params:
                 indicator_params_for_strategy[ind_type] = {'class': ind_class, 'params': current_params}

        if not indicator_params_for_strategy:
            return -1.0

        # --- 執行回測 ---
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.addstrategy(OptunaStrategy, indicator_params=indicator_params_for_strategy)
        
        data_feed = bt.feeds.PandasData(dataname=self.df_data)
        cerebro.adddata(data_feed)
        
        cerebro.broker.setcash(self.config.INITIAL_CASH)
        cerebro.broker.setcommission(commission=self.config.COMMISSION)
        
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=1)
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='pnl')

        try:
            results = cerebro.run()
            strat = results[0]
        except Exception as e:
            self.logger.debug(f"回測在 Trial {trial.number} 中出錯: {e}")
            return -1.0

        # --- 提取結果 ---
        if self.config.OPTIMIZE_METRIC == 'sharpe':
            sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
            return sharpe if sharpe is not None else -1.0
        elif self.config.OPTIMIZE_METRIC == 'sqn':
            return strat.analyzers.sqn.get_analysis().get('sqn', 0.0)
        else: # pnl
            pnl = strat.analyzers.pnl.get_analysis().pnl.net.total or 0.0
            return pnl

    def run(self) -> None:
        """執行完整參數優化流程。"""
        if not self.indicator_blueprints:
            self.logger.error("沒有從 selected_features.json 解析出任何可優化的指標。流程終止。")
            return

        self.logger.info(f"========= 參數優化流程開始 (共 {self.config.N_TRIALS} 次嘗試) =========")
        self.logger.info(f"優化目標: {self.config.OPTIMIZE_METRIC.upper()}")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.config.N_TRIALS, show_progress_bar=True)

        self.logger.info("優化完成！")
        self.logger.info(f"最佳 Trial: {study.best_trial.number}")
        self.logger.info(f"最佳分數 ({self.config.OPTIMIZE_METRIC}): {study.best_value:.4f}")
        self.logger.info("最佳參數:")
        for key, value in study.best_params.items():
            self.logger.info(f"  {key}: {value}")

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
