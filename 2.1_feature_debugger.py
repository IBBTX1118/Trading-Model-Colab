# 檔名: 2.1_feature_debugger.py
# 描述: 專用於偵錯 02_feature_engineering.py，檢查每一步的 NaN 產生情況
# 版本: 1.1 (修復版 - 移除魔法命令並增強診斷功能)

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
import sys
import warnings
from finta import TA
from collections import defaultdict

# 忽略警告信息
warnings.filterwarnings('ignore')

# ==============================================================================
#                      配置類別 (與 02 號腳本保持一致)
# ==============================================================================
class Config:
    INPUT_BASE_DIR = Path("Output_Data_Pipeline_v2/MarketData")
    OUTPUT_BASE_DIR = Path("Output_Feature_Engineering/MarketData_with_Combined_Features_v3")
    LOG_LEVEL = "INFO"
    TIMEFRAME_ORDER = ['D1', 'H4', 'H1']
    EVENTS_FILE_PATH = Path("economic_events.csv")
    GLOBAL_FACTORS_FILE_PATH = Path("global_factors.parquet")

# ==============================================================================
#                      特徵工程類別 (簡化版，專用於調試)
# ==============================================================================
class FeatureEngineerDebugger:
    """特徵工程調試器 - 逐步執行並檢查每個步驟的數據健康狀況"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.step_results = []  # 記錄每個步驟的結果
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.LOG_LEVEL.upper())
        if not logger.hasHandlers():
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(formatter)
            logger.addHandler(sh)
        return logger

    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加基礎技術分析特徵"""
        self.logger.debug("Adding base features...")
        df_finta = df.copy()
        
        # 重命名成交量欄位以配合finta
        df_finta.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        try:
            # ATR 計算
            for p in [5, 14, 20]:
                df_finta[f'ATR_{p}'] = TA.ATR(df_finta, period=p)
            
            # DMI 指標
            df_finta = df_finta.join(TA.DMI(df_finta))
            
            # 移動平均和 RSI
            periods = [10, 20, 50]
            for indicator in ['SMA', 'EMA']:
                method = getattr(TA, indicator)
                for p in periods:
                    df_finta[f'{indicator}_{p}'] = method(df_finta, period=p)
            
            # RSI
            df_finta['RSI_14'] = TA.RSI(df_finta, period=14)
            
            # 其他指標
            df_finta = df_finta.join(TA.STOCH(df_finta))
            df_finta = df_finta.join(TA.MACD(df_finta))
            df_finta = df_finta.join(TA.BBANDS(df_finta))
            
            # 價格行為特徵
            df_finta['body_size'] = abs(df_finta['close'] - df_finta['open'])
            df_finta['upper_wick'] = df_finta['high'] - df_finta[['open', 'close']].max(axis=1)
            df_finta['lower_wick'] = df_finta[['open', 'close']].min(axis=1) - df_finta['low']
            df_finta['body_vs_wick'] = df_finta['body_size'] / (df_finta['high'] - df_finta['low'] + 1e-9)
            
            # DMI 差異
            if 'DI+' in df_finta.columns and 'DI-' in df_finta.columns:
                df_finta['DMI_DIFF'] = abs(df_finta['DI+'] - df_finta['DI-'])
            
            # 標準化特徵
            if 'ATR_14' in df_finta.columns:
                atr_series = df_finta['ATR_14'] + 1e-9
                df_finta['body_size_norm'] = df_finta['body_size'] / atr_series
                if 'DMI_DIFF' in df_finta.columns:
                    df_finta['DMI_DIFF_norm'] = df_finta['DMI_DIFF'] / atr_series
            
        except Exception as e:
            self.logger.error(f"基礎特徵計算時發生錯誤: {e}")
        
        # 恢復原始欄位名
        df_finta.rename(columns={'volume': 'tick_volume'}, inplace=True)
        return df_finta

    def _add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """新增市場微結構特徵"""
        self.logger.debug("Adding market microstructure features...")
        df_out = df.copy()
        
        try:
            # 買賣壓力指標
            df_out['buying_pressure'] = (df_out['close'] - df_out['low']) / (df_out['high'] - df_out['low'] + 1e-9)
            
            # 價格效率比率
            net_change = abs(df_out['close'].diff(20))
            total_movement = df_out['close'].diff().abs().rolling(20).sum()
            df_out['efficiency_ratio'] = net_change / (total_movement + 1e-9)
            
            # 訂單流不平衡代理
            ofi_proxy = (df_out['tick_volume'] * np.sign(df_out['close'].diff())).fillna(0)
            df_out['ofi_cumsum_20'] = ofi_proxy.rolling(20).sum()
            
        except Exception as e:
            self.logger.error(f"市場微結構特徵計算時發生錯誤: {e}")
            
        return df_out

    def _add_regime_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """新增市場狀態識別特徵"""
        self.logger.debug("Adding regime detection features...")
        df_out = df.copy()
        
        try:
            # 波動率狀態
            returns = df_out['close'].pct_change()
            df_out['volatility_20p'] = returns.rolling(20).std()
            
            # Z-score 標準化
            vol_mean_60 = df_out['volatility_20p'].rolling(60).mean()
            vol_std_60 = df_out['volatility_20p'].rolling(60).std()
            df_out['vol_regime_zscore'] = (df_out['volatility_20p'] - vol_mean_60) / (vol_std_60 + 1e-9)
            
            # ADX 趨勢強度
            try:
                adx_df = TA.ADX(df_out, period=14)
                if isinstance(adx_df, pd.DataFrame) and 'ADX' in adx_df.columns:
                    df_out['adx_14'] = adx_df['ADX']
                else:
                    df_out['adx_14'] = adx_df
                
                df_out['trend_strength'] = (df_out['adx_14'] / 100).fillna(0)
            except Exception as e:
                self.logger.warning(f"ADX計算失敗，使用替代方法: {e}")
                # 使用移動平均斜率作為替代
                ma_20 = df_out['close'].rolling(20).mean()
                trend_slope = ma_20.diff(10) / ma_20.shift(10)
                df_out['trend_strength'] = abs(trend_slope).fillna(0)
                df_out['adx_14'] = df_out['trend_strength'] * 100
            
            # 市場狀態分類
            conditions = [
                (df_out['vol_regime_zscore'] > 1) & (df_out['trend_strength'] > 0.3),   # 高波動趨勢
                (df_out['vol_regime_zscore'] > 1) & (df_out['trend_strength'] <= 0.3),  # 高波動盤整
                (df_out['vol_regime_zscore'] <= 1) & (df_out['trend_strength'] > 0.3),  # 低波動趨勢
                (df_out['vol_regime_zscore'] <= 1) & (df_out['trend_strength'] <= 0.3)   # 低波動盤整
            ]
            choices = [3, 2, 1, 0]
            df_out['market_regime'] = np.select(conditions, choices, default=0)
            
        except Exception as e:
            self.logger.error(f"市場狀態識別特徵計算時發生錯誤: {e}")
            
        return df_out

    def _add_advanced_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加高級波動率特徵"""
        self.logger.debug("Adding advanced volatility features...")
        df_out = df.copy()
        
        try:
            # ATR 比率
            if 'ATR_5' in df_out.columns and 'ATR_20' in df_out.columns:
                df_out['atr_ratio_5_20'] = df_out['ATR_5'] / (df_out['ATR_20'] + 1e-9)
            
            # 上漲和下跌波動率
            returns = df_out['close'].pct_change()
            upside_returns = returns.where(returns > 0, 0)
            downside_returns = returns.where(returns < 0, 0)
            
            df_out['upside_vol_20p'] = upside_returns.rolling(20).std()
            df_out['downside_vol_20p'] = downside_returns.rolling(20).std()
            df_out['vol_asymmetry'] = df_out['upside_vol_20p'] / (df_out['downside_vol_20p'] + 1e-9)
            
            # 使用前向填充處理 NaN
            volatility_cols = ['upside_vol_20p', 'downside_vol_20p', 'vol_asymmetry']
            df_out[volatility_cols] = df_out[volatility_cols].fillna(method='ffill')
            
        except Exception as e:
            self.logger.error(f"高級波動率特徵計算時發生錯誤: {e}")
            
        return df_out

    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加價格行為特徵"""
        self.logger.debug("Adding price action features...")
        df_out = df.copy()
        
        try:
            # 相對於移動平均的位置
            if 'SMA_50' in df_out.columns and 'ATR_14' in df_out.columns:
                df_out['price_vs_sma50_norm'] = (df_out['close'] - df_out['SMA_50']) / (df_out['ATR_14'] + 1e-9)
            
            # 相對於布林帶的位置
            if all(col in df_out.columns for col in ['BB_MIDDLE', 'BB_UPPER', 'BB_LOWER']):
                bb_width = df_out['BB_UPPER'] - df_out['BB_LOWER'] + 1e-9
                df_out['price_vs_bb'] = (df_out['close'] - df_out['BB_MIDDLE']) / bb_width * 2
            
            # 距離高低點的K線數
            window = 50
            df_out[f'bars_since_{window}p_high'] = df_out['high'].rolling(window).apply(
                lambda x: len(x) - np.argmax(x) - 1, raw=True
            )
            df_out[f'bars_since_{window}p_low'] = df_out['low'].rolling(window).apply(
                lambda x: len(x) - np.argmin(x) - 1, raw=True
            )
            
        except Exception as e:
            self.logger.error(f"價格行為特徵計算時發生錯誤: {e}")
            
        return df_out

    def _add_multi_period_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加多週期特徵"""
        self.logger.debug("Adding multi-period features...")
        df_out = df.copy()
        
        try:
            for n in [5, 10, 20, 60]:
                df_out[f'returns_{n}p'] = df_out['close'].pct_change(periods=n)
                # 添加波動率特徵
                returns = df_out['close'].pct_change()
                df_out[f'volatility_{n}p'] = returns.rolling(window=n).std() * np.sqrt(n)
                
        except Exception as e:
            self.logger.error(f"多週期特徵計算時發生錯誤: {e}")
            
        return df_out

    def _add_advanced_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加高級時間特徵"""
        self.logger.debug("Adding advanced time features...")
        df_out = df.copy()
        
        try:
            hour = df_out.index.hour
            df_out['day_of_week'] = df_out.index.dayofweek
            df_out['hour_of_day'] = hour
            df_out['month_of_year'] = df_out.index.month
            
            # 交易時段特徵
            df_out['is_tokyo_session'] = ((hour >= 0) & (hour <= 8)).astype(int)
            df_out['is_london_session'] = ((hour >= 8) & (hour <= 16)).astype(int)
            df_out['is_ny_session'] = ((hour >= 13) & (hour <= 21)).astype(int)
            df_out['is_london_ny_overlap'] = ((hour >= 13) & (hour <= 16)).astype(int)
            
            # 注意：這裡不使用 pd.get_dummies，因為會產生大量稀疏特徵
            
        except Exception as e:
            self.logger.error(f"高級時間特徵計算時發生錯誤: {e}")
            
        return df_out

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加交互作用特徵"""
        self.logger.debug("Adding interaction features...")
        df_out = df.copy()
        
        try:
            # RSI 與波動率的交互作用
            if 'RSI_14' in df_out.columns and 'volatility_20p' in df_out.columns:
                df_out['RSI_x_volatility'] = (df_out['RSI_14'] - 50) * df_out['volatility_20p']
            
            # 市場狀態與趨勢的交互作用
            if 'vol_regime_zscore' in df_out.columns and 'trend_strength' in df_out.columns:
                df_out['regime_x_trend'] = df_out['vol_regime_zscore'] * df_out['trend_strength']
            
            # RSI 與價格位置的交互作用
            if 'RSI_14' in df_out.columns and 'price_vs_sma50_norm' in df_out.columns:
                df_out['rsi_x_pullback'] = (df_out['RSI_14'] - 50) * df_out['price_vs_sma50_norm']
                
        except Exception as e:
            self.logger.error(f"交互作用特徵計算時發生錯誤: {e}")
            
        return df_out

    def report_health(self, df: pd.DataFrame, step_name: str) -> pd.DataFrame:
        """報告數據健康狀況"""
        print(f"\n{'='*60}")
        print(f"🔍 執行步驟: {step_name}")
        print(f"{'='*60}")
        
        # 基本統計
        rows_total = len(df)
        nan_count = df.isnull().sum().sum()
        
        print(f"📊 基本統計:")
        print(f"   當前形狀: {df.shape}")
        print(f"   總 NaN 數量: {nan_count:,}")
        print(f"   NaN 比例: {nan_count / (df.shape[0] * df.shape[1]) * 100:.2f}%")
        
        # 檢查如果執行 dropna() 的影響
        df_dropped = df.dropna()
        rows_after_dropna = len(df_dropped)
        rows_lost = rows_total - rows_after_dropna
        
        print(f"📉 dropna() 影響分析:")
        print(f"   執行前行數: {rows_total:,}")
        print(f"   執行後行數: {rows_after_dropna:,}")
        print(f"   損失行數: {rows_lost:,} ({rows_lost/rows_total*100:.2f}%)")
        
        # 危險警告
        if rows_after_dropna == 0 and rows_total > 0:
            print(f"🚨🚨🚨 嚴重警告：此步驟後執行 dropna() 將導致數據完全清空！🚨🚨🚨")
        elif rows_lost / rows_total > 0.5:
            print(f"⚠️  警告：此步驟導致超過50%的數據包含NaN值！")
        elif rows_lost / rows_total > 0.2:
            print(f"⚠️  注意：此步驟導致超過20%的數據包含NaN值")
        else:
            print(f"✅ 數據健康狀況良好")
        
        # 分析哪些欄位包含最多NaN
        nan_by_column = df.isnull().sum()
        nan_columns = nan_by_column[nan_by_column > 0].sort_values(ascending=False)
        
        if len(nan_columns) > 0:
            print(f"\n📋 NaN 最多的前10個欄位:")
            for i, (col, nan_count) in enumerate(nan_columns.head(10).items()):
                percentage = nan_count / len(df) * 100
                print(f"   {i+1:2d}. {col}: {nan_count:,} ({percentage:.1f}%)")
        
        # 記錄結果
        step_result = {
            'step_name': step_name,
            'total_rows': rows_total,
            'total_cols': df.shape[1],
            'nan_count': nan_count,
            'nan_percentage': nan_count / (df.shape[0] * df.shape[1]) * 100,
            'rows_after_dropna': rows_after_dropna,
            'rows_lost': rows_lost,
            'data_loss_percentage': rows_lost/rows_total*100 if rows_total > 0 else 0,
            'top_nan_columns': dict(nan_columns.head(5))
        }
        self.step_results.append(step_result)
        
        return df

    def run_debug_pipeline(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """運行完整的調試流程"""
        print("🚀 開始特徵工程調試流程...")
        print(f"📊 原始數據形狀: {df_raw.shape}")
        
        # 複製原始數據
        df_processed = df_raw.copy()
        
        # 逐步執行特徵工程並檢查
        steps = [
            ("原始數據", lambda x: x),
            ("基礎特徵", self._add_base_features),
            ("市場微結構特徵", self._add_market_microstructure_features),
            ("市場狀態識別特徵", self._add_regime_detection_features),
            ("高級波動率特徵", self._add_advanced_volatility_features),
            ("價格行為特徵", self._add_price_action_features),
            ("多週期特徵", self._add_multi_period_features),
            ("高級時間特徵", self._add_advanced_time_features),
            ("交互作用特徵", self._add_interaction_features),
        ]
        
        for step_name, step_func in steps:
            try:
                if step_name != "原始數據":
                    df_processed = step_func(df_processed)
                df_processed = self.report_health(df_processed, step_name)
            except Exception as e:
                print(f"❌ 步驟 '{step_name}' 執行失敗: {e}")
                import traceback
                traceback.print_exc()
                break
        
        return df_processed

    def generate_summary_report(self):
        """生成總結報告"""
        print(f"\n{'='*80}")
        print("📈 特徵工程調試總結報告")
        print(f"{'='*80}")
        
        if not self.step_results:
            print("❌ 沒有步驟結果可供分析")
            return
        
        print(f"{'步驟名稱':<20} {'行數':<10} {'列數':<8} {'NaN%':<8} {'損失%':<8} {'狀態'}")
        print("-" * 80)
        
        for result in self.step_results:
            status = "🚨危險" if result['rows_after_dropna'] == 0 else \
                    "⚠️警告" if result['data_loss_percentage'] > 20 else \
                    "✅正常"
            
            print(f"{result['step_name']:<20} "
                  f"{result['total_rows']:<10,} "
                  f"{result['total_cols']:<8} "
                  f"{result['nan_percentage']:<7.1f}% "
                  f"{result['data_loss_percentage']:<7.1f}% "
                  f"{status}")
        
        # 找出問題步驟
        problem_steps = [r for r in self.step_results if r['data_loss_percentage'] > 50]
        
        if problem_steps:
            print(f"\n⚠️  發現 {len(problem_steps)} 個問題步驟:")
            for step in problem_steps:
                print(f"   • {step['step_name']}: 損失 {step['data_loss_percentage']:.1f}% 的數據")
                if step['top_nan_columns']:
                    print(f"     主要NaN欄位: {list(step['top_nan_columns'].keys())[:3]}")

def main():
    """主函數"""
    print("🚀 特徵工程調試工具 v1.1")
    print("="*60)
    
    # --- 設定 ---
    MARKET_NAME = "EURUSD_sml_H4"
    RAW_DATA_PATH = Path("Output_Data_Pipeline_v2/MarketData")
    
    print(f"🎯 目標市場: {MARKET_NAME}")
    print(f"📁 數據路徑: {RAW_DATA_PATH}")
    
    # --- 載入原始數據 ---
    print(f"\n📂 載入原始數據...")
    
    raw_market_folder = MARKET_NAME.split('_')[0] + "_" + MARKET_NAME.split('_')[1]
    raw_data_file = RAW_DATA_PATH / raw_market_folder / f"{MARKET_NAME}.parquet"
    
    if not raw_data_file.exists():
        print(f"❌ 數據檔案不存在: {raw_data_file}")
        print("💡 請確認路徑設定是否正確")
        return
    
    try:
        df_raw = pd.read_parquet(raw_data_file)
        print(f"✅ 成功載入原始數據: {df_raw.shape}")
        print(f"   時間範圍: {df_raw.index.min()} 到 {df_raw.index.max()}")
        print(f"   欄位: {list(df_raw.columns)}")
    except Exception as e:
        print(f"❌ 數據載入失敗: {e}")
        return
    
    # --- 執行調試流程 ---
    config = Config()
    debugger = FeatureEngineerDebugger(config)
    
    try:
        df_final = debugger.run_debug_pipeline(df_raw)
        debugger.generate_summary_report()
        
        print(f"\n🎉 調試流程完成！")
        print(f"   最終數據形狀: {df_final.shape}")
        
    except Exception as e:
        print(f"❌ 調試流程執行失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  用戶中斷執行")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程式執行時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
