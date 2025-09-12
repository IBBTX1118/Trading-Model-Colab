# 檔名: 2_feature_engineering_enhanced.py
# 版本: 8.0 (整合偵錯版)
# 描述: 執行完整的特徵工程後，自動對一個樣本市場進行數據健康度診斷。

import logging
import sys
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
from finta import TA
from collections import defaultdict

# ==============================================================================
#                      1. 配置類別
# ==============================================================================
class Config:
    INPUT_BASE_DIR = Path("Output_Data_Pipeline_v2/MarketData")
    OUTPUT_BASE_DIR = Path("Output_Feature_Engineering/MarketData_with_Combined_Features_v3")
    LOG_LEVEL = "INFO"
    TIMEFRAME_ORDER = ['D1', 'H4', 'H1']
    EVENTS_FILE_PATH = Path("economic_events.csv")
    GLOBAL_FACTORS_FILE_PATH = Path("global_factors.parquet")
    DEBUG_MARKET_SAMPLE = "EURUSD_sml_H4" # 指定一個用於自動偵錯的樣本市場

# ==============================================================================
#                      2. 特徵工程主類別
# ==============================================================================
class FeatureEngineer:
    # ... 此處的 FeatureEngineer class 的所有內容與您提供的
    # ... ibbtx1118/.../2_feature_engineering.py 完全相同，為了簡潔此處省略。
    # ... 請將您原本的 FeatureEngineer class 內容完整複製到這裡。
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

    def _add_global_factor_features(self, df: pd.DataFrame, factors_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """添加全局因子特徵 (DXY, VIX等)"""
        if factors_df is None or factors_df.empty:
            return df
        df_merged = pd.merge_asof(df.sort_index(), factors_df, left_index=True, right_index=True, direction='backward')
        if 'DXY_close' in df_merged.columns:
            df_merged['DXY_return_1d'] = df_merged['DXY_close'].pct_change(periods=1)
            df_merged['DXY_return_5d'] = df_merged['DXY_close'].pct_change(periods=5)
        if 'VIX_close' in df_merged.columns:
            df_merged['VIX_SMA_20'] = df_merged['VIX_close'].rolling(window=20).mean()
            if 'VIX_SMA_20' in df_merged.columns and df_merged['VIX_SMA_20'].notna().any():
                 df_merged['VIX_vs_SMA20'] = (df_merged['VIX_close'] - df_merged['VIX_SMA_20']) / (df_merged['VIX_SMA_20'] + 1e-9)
        self.logger.debug("Added global factor features (DXY, VIX).")
        return df_merged

    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加基礎技術分析特徵"""
        df_finta = df.copy()
        df_finta.rename(columns={'tick_volume': 'volume'}, inplace=True)
        for p in [5, 14, 20]: df_finta[f'ATR_{p}'] = TA.ATR(df_finta, period=p)
        df_finta = df_finta.join(TA.DMI(df_finta))
        periods = [10, 20, 50]
        for indicator in ['SMA', 'EMA', 'RSI']:
            method = getattr(TA, indicator)
            if indicator == 'RSI': df_finta[f'{indicator}_14'] = method(df_finta, period=14)
            else:
                for p in periods: df_finta[f'{indicator}_{p}'] = method(df_finta, period=p)
        df_finta = df_finta.join(TA.STOCH(df_finta)); df_finta = df_finta.join(TA.MACD(df_finta)); df_finta = df_finta.join(TA.BBANDS(df_finta))
        df_finta['body_size'] = abs(df_finta['close'] - df_finta['open'])
        df_finta['upper_wick'] = df_finta['high'] - df_finta[['open', 'close']].max(axis=1)
        df_finta['lower_wick'] = df_finta[['open', 'close']].min(axis=1) - df_finta['low']
        df_finta['body_vs_wick'] = df_finta['body_size'] / (df_finta['high'] - df_finta['low'] + 1e-9)
        if 'DI+' in df_finta.columns and 'DI-' in df_finta.columns: df_finta['DMI_DIFF'] = abs(df_finta['DI+'] - df_finta['DI-'])
        if 'ATR_14' in df_finta.columns:
            atr_series = df_finta['ATR_14'] + 1e-9
            df_finta['body_size_norm'] = df_finta['body_size'] / atr_series
            if 'DMI_DIFF' in df_finta.columns: df_finta['DMI_DIFF_norm'] = df_finta['DMI_DIFF'] / atr_series
        df_finta.rename(columns={'volume': 'tick_volume'}, inplace=True)
        return df_finta

    def _add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """新增市場微結構特徵,以捕捉K棒內的買賣力道"""
        df_out = df.copy()
        df_out['buying_pressure'] = (df_out['close'] - df_out['low']) / (df_out['high'] - df_out['low'] + 1e-9)
        net_change = abs(df_out['close'].diff(20)); total_movement = df_out['close'].diff().abs().rolling(20).sum()
        df_out['efficiency_ratio'] = net_change / (total_movement + 1e-9)
        ofi_proxy = (df_out['tick_volume'] * np.sign(df_out['close'].diff())).fillna(0)
        df_out['ofi_cumsum_20'] = ofi_proxy.rolling(20).sum()
        return df_out

    def _add_regime_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """新增市場狀態識別特徵,以判斷趨勢/盤整及波動狀態"""
        df_out = df.copy()
        returns = df_out['close'].pct_change(); df_out['volatility_20p'] = returns.rolling(20).std()
        vol_mean_60 = df_out['volatility_20p'].rolling(60).mean(); vol_std_60 = df_out['volatility_20p'].rolling(60).std()
        df_out['vol_regime_zscore'] = (df_out['volatility_20p'] - vol_mean_60) / (vol_std_60 + 1e-9)
        try:
            adx_df = TA.ADX(df_out, period=14)
            df_out['adx_14'] = adx_df['ADX'] if isinstance(adx_df, pd.DataFrame) else adx_df
            df_out['trend_strength'] = (df_out['adx_14'] / 100).fillna(0)
        except Exception as e:
            self.logger.warning(f"ADX calculation failed: {e}. Using alternative method.")
            ma_20 = df_out['close'].rolling(20).mean(); trend_slope = ma_20.diff(10) / ma_20.shift(10)
            df_out['trend_strength'] = abs(trend_slope).fillna(0); df_out['adx_14'] = df_out['trend_strength'] * 100
        conditions = [(df_out['vol_regime_zscore'] > 1) & (df_out['trend_strength'] > 0.3), (df_out['vol_regime_zscore'] > 1) & (df_out['trend_strength'] <= 0.3),
                      (df_out['vol_regime_zscore'] <= 1) & (df_out['trend_strength'] > 0.3), (df_out['vol_regime_zscore'] <= 1) & (df_out['trend_strength'] <= 0.3)]
        choices = [3, 2, 1, 0]; df_out['market_regime'] = np.select(conditions, choices, default=0)
        return df_out

    def _add_advanced_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加高級時間特徵"""
        df_out = df.copy(); hour = df_out.index.hour
        df_out['day_of_week'] = df_out.index.dayofweek; df_out['hour_of_day'] = hour; df_out['month_of_year'] = df_out.index.month
        df_out['is_tokyo_session'] = ((hour >= 0) & (hour <= 8)).astype(int); df_out['is_london_session'] = ((hour >= 8) & (hour <= 16)).astype(int)
        df_out['is_ny_session'] = ((hour >= 13) & (hour <= 21)).astype(int); df_out['is_london_ny_overlap'] = ((hour >= 13) & (hour <= 16)).astype(int)
        df_out = pd.get_dummies(df_out, columns=['day_of_week', 'hour_of_day'], prefix=['dow', 'hour'])
        return df_out

    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """增強價格行為特徵"""
        df_out = df.copy()
        if 'SMA_50' in df_out.columns and 'ATR_14' in df_out.columns: df_out['price_vs_sma50_norm'] = (df_out['close'] - df_out['SMA_50']) / (df_out['ATR_14'] + 1e-9)
        if all(col in df_out.columns for col in ['BB_MIDDLE', 'BB_UPPER', 'BB_LOWER']):
            bb_width = df_out['BB_UPPER'] - df_out['BB_LOWER'] + 1e-9
            df_out['price_vs_bb'] = (df_out['close'] - df_out['BB_MIDDLE']) / bb_width * 2
        window = 50
        df_out[f'bars_since_{window}p_high'] = df_out['high'].rolling(window).apply(lambda x: len(x) - np.argmax(x) - 1, raw=True)
        df_out[f'bars_since_{window}p_low'] = df_out['low'].rolling(window).apply(lambda x: len(x) - np.argmin(x) - 1, raw=True)
        return df_out

    def _add_advanced_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """增強波動率特徵"""
        df_out = df.copy()
        if 'ATR_5' in df_out.columns and 'ATR_20' in df_out.columns: df_out['atr_ratio_5_20'] = df_out['ATR_5'] / (df_out['ATR_20'] + 1e-9)
        returns = df_out['close'].pct_change()
        upside_returns, downside_returns = returns.where(returns > 0, 0), returns.where(returns < 0, 0)
        df_out['upside_vol_20p'] = upside_returns.rolling(20).std()
        df_out['downside_vol_20p'] = abs(downside_returns).rolling(20).std()
        df_out['vol_asymmetry'] = df_out['upside_vol_20p'] / (df_out['downside_vol_20p'] + 1e-9)
        df_out[['upside_vol_20p', 'downside_vol_20p', 'vol_asymmetry']] = df_out[['upside_vol_20p', 'downside_vol_20p', 'vol_asymmetry']].fillna(method='ffill')
        return df_out

    def _add_multi_period_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加多週期特徵"""
        df_out = df.copy()
        for n in [5, 10, 20, 60]:
            df_out[f'returns_{n}p'] = df_out['close'].pct_change(periods=n)
            returns = df_out['close'].pct_change()
            df_out[f'volatility_{n}p'] = returns.rolling(window=n).std() * np.sqrt(n)
        return df_out

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """增強互動特徵"""
        df_out = df.copy()
        if 'RSI_14' in df_out.columns and 'volatility_20p' in df_out.columns: df_out['RSI_x_volatility'] = (df_out['RSI_14'] - 50) * df_out['volatility_20p']
        if 'vol_regime_zscore' in df_out.columns and 'trend_strength' in df_out.columns: df_out['regime_x_trend'] = df_out['vol_regime_zscore'] * df_out['trend_strength']
        if 'RSI_14' in df_out.columns and 'price_vs_sma50_norm' in df_out.columns: df_out['rsi_x_pullback'] = (df_out['RSI_14'] - 50) * df_out['price_vs_sma50_norm']
        if 'vol_asymmetry' in df_out.columns and 'market_regime' in df_out.columns: df_out['asymmetry_x_regime'] = df_out['vol_asymmetry'] * df_out['market_regime']
        return df_out

    def _add_event_features(self, df: pd.DataFrame, events_df: Optional[pd.DataFrame], symbol: str) -> pd.DataFrame:
        """添加經濟事件特徵"""
        df_out = df.copy()
        if events_df is None or events_df.empty: df_out['time_to_next_event'], df_out['next_event_importance'] = 999, 0; return df_out
        currency1, currency2 = symbol[:3].upper(), symbol[3:6].upper()
        relevant_events = events_df[events_df['currency'].isin([currency1, currency2])].copy()
        if relevant_events.empty: df_out['time_to_next_event'], df_out['next_event_importance'] = 999, 0; return df_out
        relevant_events.reset_index(inplace=True)
        df_merged = pd.merge_asof(left=df_out.sort_index(), right=relevant_events.sort_values('timestamp_utc'), left_index=True, right_on='timestamp_utc', direction='forward')
        df_merged.index = df_out.index
        time_diff = df_merged['timestamp_utc'] - df_out.index
        df_out['time_to_next_event'] = time_diff.dt.total_seconds() / 3600; df_out['next_event_importance'] = df_merged['importance_val']
        df_out.fillna({'time_to_next_event': 999, 'next_event_importance': 0}, inplace=True)
        return df_out

    def process_symbol_group(self, symbol: str, paths: Dict[str, Path], events_df: Optional[pd.DataFrame], factors_df: Optional[pd.DataFrame]):
        self.logger.info(f"--- 開始處理商品組: {symbol} ---"); dataframes = {}
        for tf in self.config.TIMEFRAME_ORDER:
            if tf in paths:
                self.logger.info(f"[{symbol}] 正在處理 {tf} 數據...")
                df = pd.read_parquet(paths[tf])
                try:
                    df = self._add_global_factor_features(df, factors_df); df = self._add_event_features(df, events_df, symbol); df = self._add_base_features(df)
                    df = self._add_market_microstructure_features(df); df = self._add_regime_detection_features(df); df = self._add_advanced_volatility_features(df)
                    df = self._add_price_action_features(df); df = self._add_multi_period_features(df); df = self._add_advanced_time_features(df); df = self._add_interaction_features(df)
                    if tf == 'D1': df['is_uptrend'] = (df['close'] > df['SMA_50']).astype(int)
                    dataframes[tf] = df
                except Exception as e: self.logger.error(f"[{symbol}/{tf}] 特徵計算過程中發生錯誤: {e}"); continue
        final_dataframes = {}; cols_to_drop = ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread']
        if 'D1' in dataframes: final_dataframes['D1'] = dataframes['D1']
        if 'H4' in dataframes and 'D1' in dataframes:
            df_d1_renamed = dataframes['D1'].rename(columns=lambda c: f"D1_{c}" if c not in cols_to_drop else c)
            final_dataframes['H4'] = pd.merge_asof(dataframes['H4'].sort_index(), df_d1_renamed.drop(columns=cols_to_drop, errors='ignore'), left_index=True, right_index=True, direction='backward')
        elif 'H4' in dataframes: final_dataframes['H4'] = dataframes['H4']
        if 'H1' in dataframes and 'H4' in final_dataframes:
            df_h4_renamed = final_dataframes['H4'].rename(columns=lambda c: f"H4_{c}" if c not in cols_to_drop else c)
            final_dataframes['H1'] = pd.merge_asof(dataframes['H1'].sort_index(), df_h4_renamed.drop(columns=cols_to_drop, errors='ignore'), left_index=True, right_index=True, direction='backward')
        elif 'H1' in dataframes: final_dataframes['H1'] = dataframes['H1']
        for tf, df_final in final_dataframes.items():
            df_final.replace([np.inf, -np.inf], np.nan, inplace=True); rows_before = len(df_final)
            df_final.fillna(method='ffill', inplace=True); df_final.dropna(inplace=True); rows_after = len(df_final)
            if rows_before > rows_after: self.logger.info(f"[{symbol}/{tf}] 清理了 {rows_before - rows_after} 行初始無效數據。")
            if rows_after == 0 and rows_before > 0: self.logger.error(f"[{symbol}/{tf}] 嚴重錯誤：數據被完全清空！"); continue
            try:
                relative_path = paths[tf].relative_to(self.config.INPUT_BASE_DIR); output_path = self.config.OUTPUT_BASE_DIR / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True); df_final.to_parquet(output_path)
                self.logger.info(f"[{symbol}/{tf}] 已儲存綜合特徵檔案到: {output_path} ({df_final.shape[0]:,} 行, {df_final.shape[1]} 列)")
            except Exception as e: self.logger.error(f"[{symbol}/{tf}] 保存檔案時發生錯誤: {e}")

    def run(self):
        self.logger.info(f"========= 綜合特徵工程流程開始 (v8.0) =========")
        events_df, factors_df = None, None
        try:
            events_df = pd.read_csv(self.config.EVENTS_FILE_PATH, index_col='timestamp_utc', parse_dates=True); events_df.index = pd.to_datetime(events_df.index, utc=True)
            self.logger.info(f"成功讀取事件數據: {len(events_df)} 筆")
        except FileNotFoundError: self.logger.warning(f"警告：找不到事件檔案 {self.config.EVENTS_FILE_PATH}！")
        try:
            factors_df = pd.read_parquet(self.config.GLOBAL_FACTORS_FILE_PATH); factors_df.index = pd.to_datetime(factors_df.index, utc=True)
            self.logger.info(f"成功讀取全局因子數據: {factors_df.shape}")
        except FileNotFoundError: self.logger.warning(f"警告：找不到全局因子檔案 {self.config.GLOBAL_FACTORS_FILE_PATH}！")
        input_files = list(self.config.INPUT_BASE_DIR.rglob("*.parquet"))
        if not input_files: self.logger.warning("在輸入目錄中沒有找到任何 Parquet 檔案。"); return
        symbol_groups = defaultdict(dict)
        for file_path in input_files:
            try:
                parts = file_path.stem.split('_'); symbol, timeframe = '_'.join(parts[:-1]), parts[-1]
                if timeframe in self.config.TIMEFRAME_ORDER: symbol_groups[symbol][timeframe] = file_path
            except IndexError: self.logger.warning(f"無法解析檔名: {file_path.name}，已跳過。")
        self.logger.info(f"共找到 {len(symbol_groups)} 個商品組需要處理。")
        for symbol, paths in symbol_groups.items():
            try: self.process_symbol_group(symbol, paths, events_df, factors_df)
            except Exception as e: self.logger.error(f"處理商品組 {symbol} 時發生嚴重錯誤: {e}")
        self.logger.info(f"========= 特徵工程流程結束 =========")

# ==============================================================================
#                      3. 特徵偵錯器類別
# ==============================================================================
class FeatureDebugger:
    """用於分析特徵工程後數據健康狀況的偵錯器"""
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        # 確保日誌記錄器已設定
        if not self.logger.hasHandlers():
            self.logger.setLevel(self.config.LOG_LEVEL.upper())
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

    def run_health_check(self, market_name: str):
        """對指定的市場樣本執行健康度檢查"""
        self.logger.info(f"\n{'='*80}\n🚀 開始對樣本市場 [{market_name}] 進行特徵健康度檢查...\n{'='*80}")
        
        market_folder = "_".join(market_name.split('_')[:2])
        data_file = self.config.OUTPUT_BASE_DIR / market_folder / f"{market_name}.parquet"

        if not data_file.exists():
            self.logger.error(f"❌ 偵錯失敗：找不到特徵檔案: {data_file}")
            return

        try:
            df = pd.read_parquet(data_file)
            self.logger.info(f"✅ 成功載入特徵數據: {df.shape}")
        except Exception as e:
            self.logger.error(f"❌ 偵錯失敗：讀取特徵檔案時出錯: {e}")
            return
            
        # --- 健康度分析 ---
        rows_total = len(df)
        nan_count = df.isnull().sum().sum()
        inf_count = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
        
        print("\n📊 --- 最終數據健康度報告 ---")
        print(f"   總行數: {rows_total:,}")
        print(f"   總特徵數: {df.shape[1]}")
        
        if nan_count > 0: print(f"   ⚠️  NaN (缺失值) 總數: {nan_count:,}")
        else: print(f"   ✅ NaN (缺失值) 總數: 0")
            
        if inf_count > 0: print(f"   ⚠️  Infinite (無限值) 總數: {inf_count:,}")
        else: print(f"   ✅ Infinite (無限值) 總數: 0")
        
        # 檢查是否存在完全為空或全為NaN的欄位
        empty_cols = [col for col in df.columns if df[col].isnull().all()]
        if empty_cols:
            self.logger.warning(f"🚨 嚴重警告：以下欄位完全為空 (全是NaN)，請檢查其計算邏輯: {empty_cols}")
        else:
            self.logger.info("✅ 所有欄位都包含有效數據。")
            
        print(f"{'='*80}\n🚀 健康度檢查完畢。\n{'='*80}")

# ==============================================================================
#                      4. 主執行區塊
# ==============================================================================
if __name__ == "__main__":
    try:
        config = Config()
        
        # --- 步驟 1: 執行特徵工程 ---
        engineer = FeatureEngineer(config)
        engineer.run()
        
        # --- 步驟 2: 執行自動偵錯 ---
        debugger = FeatureDebugger(config)
        debugger.run_health_check(market_name=config.DEBUG_MARKET_SAMPLE)
        
    except Exception as e:
        logging.critical(f"腳本執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
        sys.exit(1)
