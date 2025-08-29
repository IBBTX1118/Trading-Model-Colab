# 檔名: 02.5_feature_debugger.py
# 描述: 專用於偵錯 02_feature_engineering.py，檢查每一步的 NaN 產生情況

%matplotlib inline
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
import sys

# --- 複製 02_feature_engineering.py (v7.0) 的完整 FeatureEngineer 類別 ---
# 為了方便，我們直接將整個類別複製到這裡
# (注意：請確保這裡的程式碼和您最新的 02 號腳本完全一致)
from finta import TA
from collections import defaultdict

class Config:
    INPUT_BASE_DIR = Path("Output_Data_Pipeline_v2/MarketData")
    # ... (其餘 Config 內容與 02 號腳本相同)

class FeatureEngineer:
    # ... (請將您最新的 02_feature_engineering.py v7.0 的完整 FeatureEngineer 類別貼在這裡)
    # ... (從 class FeatureEngineer: 到 class 結尾的所有程式碼)
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.LOG_LEVEL.upper())
        if not logger.hasHandlers():
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(formatter)
            logger.addHandler(sh)
        return logger

    # ... (貼上所有 _add_..._features 方法) ...
    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_finta = df.copy()
        df_finta.rename(columns={'tick_volume': 'volume'}, inplace=True)
        for p in [5, 14, 20]:
            df_finta[f'ATR_{p}'] = TA.ATR(df_finta, period=p)
        df_finta = df_finta.join(TA.DMI(df_finta))
        periods = [10, 20, 50]
        for indicator in ['SMA', 'EMA', 'RSI']:
            method = getattr(TA, indicator)
            if indicator == 'RSI':
                df_finta[f'{indicator}_14'] = method(df_finta, period=14)
            else:
                for p in periods: df_finta[f'{indicator}_{p}'] = method(df_finta, period=p)
        df_finta = df_finta.join(TA.STOCH(df_finta)); df_finta = df_finta.join(TA.MACD(df_finta)); df_finta = df_finta.join(TA.BBANDS(df_finta))
        df_finta['body_size'] = abs(df_finta['close'] - df_finta['open'])
        if 'DI+' in df_finta.columns and 'DI-' in df_finta.columns:
            df_finta['DMI_DIFF'] = abs(df_finta['DI+'] - df_finta['DI-'])
        if 'ATR_14' in df_finta.columns:
            df_finta['body_size_norm'] = df_finta['body_size'] / (df_finta['ATR_14'] + 1e-9)
        df_finta.rename(columns={'volume': 'tick_volume'}, inplace=True)
        return df_finta
    
    def _add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        df_out['buying_pressure'] = (df_out['close'] - df_out['low']) / (df_out['high'] - df_out['low'] + 1e-9)
        net_change = abs(df_out['close'].diff(20))
        total_movement = df_out['close'].diff().abs().rolling(20).sum()
        df_out['efficiency_ratio'] = net_change / (total_movement + 1e-9)
        ofi_proxy = (df_out['tick_volume'] * np.sign(df_out['close'].diff())).fillna(0)
        df_out['ofi_cumsum_20'] = ofi_proxy.rolling(20).sum()
        return df_out

    def _add_regime_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        returns = df_out['close'].pct_change()
        df_out['volatility_20p'] = returns.rolling(20).std()
        vol_mean_60, vol_std_60 = df_out['volatility_20p'].rolling(60).mean(), df_out['volatility_20p'].rolling(60).std()
        df_out['vol_regime_zscore'] = (df_out['volatility_20p'] - vol_mean_60) / (vol_std_60 + 1e-9)
        adx_df = TA.ADX(df_out, period=14)
        df_out['adx_14'] = adx_df['ADX'] if isinstance(adx_df, pd.DataFrame) else adx_df
        df_out['trend_strength'] = (df_out['adx_14'] / 100).fillna(0)
        conditions = [
            (df_out['vol_regime_zscore'] > 1) & (df_out['trend_strength'] > 0.3),
            (df_out['vol_regime_zscore'] > 1) & (df_out['trend_strength'] <= 0.3),
            (df_out['vol_regime_zscore'] <= 1) & (df_out['trend_strength'] > 0.3),
            (df_out['vol_regime_zscore'] <= 1) & (df_out['trend_strength'] <= 0.3)
        ]
        df_out['market_regime'] = np.select(conditions, [3, 2, 1, 0], default=0)
        return df_out

    def _add_advanced_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if 'ATR_5' in df_out.columns and 'ATR_20' in df_out.columns:
            df_out['atr_ratio_5_20'] = df_out['ATR_5'] / (df_out['ATR_20'] + 1e-9)
        returns = df_out['close'].pct_change()
        df_out['upside_vol_20p'] = returns[returns > 0].rolling(20).std()
        df_out['downside_vol_20p'] = returns[returns < 0].rolling(20).std()
        df_out['vol_asymmetry'] = df_out['upside_vol_20p'] / (df_out['downside_vol_20p'] + 1e-9)
        df_out[['upside_vol_20p', 'downside_vol_20p', 'vol_asymmetry']] = df_out[['upside_vol_20p', 'downside_vol_20p', 'vol_asymmetry']].fillna(method='ffill')
        return df_out
        
    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if 'SMA_50' in df_out.columns and 'ATR_14' in df_out.columns:
            df_out['price_vs_sma50_norm'] = (df_out['close'] - df_out['SMA_50']) / (df_out['ATR_14'] + 1e-9)
        if 'BB_MIDDLE' in df_out.columns and 'BB_UPPER' in df_out.columns and 'BB_LOWER' in df_out.columns:
            df_out['price_vs_bb'] = (df_out['close'] - df_out['BB_MIDDLE']) / (df_out['BB_UPPER'] - df_out['BB_LOWER'] + 1e-9) * 2
        window = 50
        df_out[f'bars_since_{window}p_high'] = df_out['high'].rolling(window).apply(lambda x: len(x) - np.argmax(x) - 1, raw=True)
        df_out[f'bars_since_{window}p_low'] = df_out['low'].rolling(window).apply(lambda x: len(x) - np.argmin(x) - 1, raw=True)
        return df_out
        
    def _add_multi_period_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for n in [5, 10, 20, 60]:
            df_out[f'returns_{n}p'] = df_out['close'].pct_change(periods=n)
        return df_out

    def _add_advanced_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        hour = df_out.index.hour
        df_out['day_of_week'] = df_out.index.dayofweek
        df_out['hour_of_day'] = hour
        df_out = pd.get_dummies(df_out, columns=['day_of_week', 'hour_of_day'], prefix=['dow', 'hour'])
        return df_out

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if 'RSI_14' in df_out.columns and 'volatility_20p' in df_out.columns:
            df_out['RSI_x_volatility'] = (df_out['RSI_14'] - 50) * df_out['volatility_20p']
        if 'vol_regime_zscore' in df_out.columns and 'trend_strength' in df_out.columns:
            df_out['regime_x_trend'] = df_out['vol_regime_zscore'] * df_out['trend_strength']
        if 'RSI_14' in df_out.columns and 'price_vs_sma50_norm' in df_out.columns:
            df_out['rsi_x_pullback'] = (df_out['RSI_14'] - 50) * df_out['price_vs_sma50_norm']
        return df_out

# --- 偵錯主流程 ---
print("🚀 開始數據管道偵錯流程...")

# --- 設定 ---
MARKET_NAME = "EURUSD_sml_H4"
RAW_DATA_PATH = Path("Output_Data_Pipeline_v2/MarketData")

# --- 載入最原始的數據 ---
raw_market_folder = MARKET_NAME.split('_')[0] + "_" + MARKET_NAME.split('_')[1]
raw_data_file = RAW_DATA_PATH / raw_market_folder / f"{MARKET_NAME}.parquet"
df_raw = pd.read_parquet(raw_data_file)
print(f"✅ 成功載入原始數據: {df_raw.shape}")

# 實例化 FeatureEngineer
fe = FeatureEngineer(Config())

# 定義一個報告函數
def report_health(df, step_name):
    print(f"\n--- 執行步驟: {step_name} ---")
    nan_count = df.isnull().sum().sum()
    rows_before = len(df)
    df_dropped = df.dropna()
    rows_after = len(df_dropped)
    print(f"  - 當前形狀: {df.shape}")
    print(f"  - NaN 總數: {nan_count:,}")
    print(f"  - 若執行 dropna()，數據將從 {rows_before:,} 筆變為 {rows_after:,} 筆 (損失 {rows_before - rows_after:,} 筆)")
    if rows_after == 0 and rows_before > 0:
        print("  - 🚨🚨🚨 警告：此步驟後若 dropna() 將導致數據清空！🚨🚨🚨")
    return df

# --- 按順序執行每一步特徵工程並檢查 ---
df_processed = df_raw.copy()

df_processed = fe._add_base_features(df_processed)
df_processed = report_health(df_processed, "_add_base_features")

df_processed = fe._add_market_microstructure_features(df_processed)
df_processed = report_health(df_processed, "_add_market_microstructure_features")

df_processed = fe._add_regime_detection_features(df_processed)
df_processed = report_health(df_processed, "_add_regime_detection_features")

df_processed = fe._add_advanced_volatility_features(df_processed)
df_processed = report_health(df_processed, "_add_advanced_volatility_features")

df_processed = fe._add_price_action_features(df_processed)
df_processed = report_health(df_processed, "_add_price_action_features")

df_processed = fe._add_multi_period_features(df_processed)
df_processed = report_health(df_processed, "_add_multi_period_features")

df_processed = fe._add_advanced_time_features(df_processed)
df_processed = report_health(df_processed, "_add_advanced_time_features")

df_processed = fe._add_interaction_features(df_processed)
df_processed = report_health(df_processed, "_add_interaction_features")

print("\n🎉 偵錯流程結束！請檢查上面的報告，找出導致數據清空的步驟。")
