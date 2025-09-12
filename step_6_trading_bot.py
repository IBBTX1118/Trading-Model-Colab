# 檔名: step_6_trading_bot.py
# 版本: 1.4 (最終命名版)
# 描述: 用於連接 OANDA MT5 模擬帳戶並執行 H1 交易策略的機器人模板。

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import sys
from pathlib import Path
import json
import joblib
from typing import Dict, Optional

# ★★★ 核心修正：從新的檔名 step_2_feature_engineering.py 導入所需類別 ★★★
from step_2_feature_engineering import FeatureEngineer, Config as FeatureConfig

# ==============================================================================
#                      1. 設定區塊
# ==============================================================================
# --- 帳戶設定 (請務必填寫您的真實資訊) ---
MT5_LOGIN = 1600014313                          # ★★★ 請替換為您的 OANDA MT5 模擬帳戶登入號 ★★★
MT5_PASSWORD = "DjQlK*1x"                  # ★★★ 請替換為您的帳戶密碼 ★★★
MT5_SERVER = "OANDA-Demo-1"
MT5_PATH = r"C:\Program Files\OANDA MetaTrader 5\terminal64.exe" # ★★★ 請替換為您電腦上 MT5 的完整路徑 ★★★

# --- 策略設定 (已為 H1 預先配置) ---
SYMBOL = "EURUSD.sml" # 要交易的商品 (請確保與您 MT5 中的名稱一致)
TIMEFRAME = mt5.TIMEFRAME_H1
TIMEFRAME_STR = "H1"

# --- 路徑設定 ---
ML_OUTPUT_PATH = Path("Output_ML_Pipeline")
FEATURES_FILE = ML_OUTPUT_PATH / f"selected_features_{SYMBOL}_{TIMEFRAME_STR}.json"
FINAL_MODEL_FILE = ML_OUTPUT_PATH / f"final_model_{SYMBOL}_{TIMEFRAME_STR}.joblib" 
BEST_PARAMS_FILE = ML_OUTPUT_PATH / f"{SYMBOL}_{TIMEFRAME_STR}_best_params_binary_lgbm.json"

# --- 交易參數 ---
RISK_PER_TRADE = 0.015 # 每次交易承擔的風險

# ==============================================================================
#                      2. 交易機器人類別 (其餘不變)
# ==============================================================================
def setup_logger():
    """設定日誌記錄器"""
    logger = logging.getLogger("TradingBot")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler('trading_bot.log', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger

class TradingBot:
    def __init__(self, logger):
        self.logger = logger
        self.model = None
        self.features = None
        self.best_params = None
        feature_config = FeatureConfig()
        self.feature_engineer = FeatureEngineer(feature_config)

    def connect_mt5(self):
        """連接到 MT5"""
        self.logger.info("正在連接到 MetaTrader 5...")
        if not mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            self.logger.error(f"MT5 初始化失敗, 錯誤代碼 = {mt5.last_error()}"); return False
        self.logger.info(f"✅ MT5 連接成功! 版本: {mt5.version()}"); return True

    def load_model_and_params(self):
        """載入最終模型、特徵列表和最佳策略參數"""
        try:
            self.logger.info(f"正在從 {FINAL_MODEL_FILE} 載入最終模型...")
            self.model = joblib.load(FINAL_MODEL_FILE)
            self.logger.info(f"正在從 {FEATURES_FILE} 載入特徵列表...")
            with open(FEATURES_FILE, 'r') as f:
                self.features = json.load(f)['selected_features']
            self.logger.info(f"正在從 {BEST_PARAMS_FILE} 載入最佳參數...")
            with open(BEST_PARAMS_FILE, 'r') as f:
                self.best_params = json.load(f)['folds_data'][-1] 
            self.logger.info(f"✅ 模型、{len(self.features)} 個特徵及策略參數載入成功。")
            return True
        except Exception as e:
            self.logger.error(f"❌ 載入檔案時發生錯誤: {e}", exc_info=True); return False
    
    def get_market_data(self, count=300) -> Optional[Dict[str, pd.DataFrame]]:
        """獲取最新的 H1, H4, D1 數據"""
        try:
            dfs = {}
            timeframes = {'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1}
            for tf_str, tf_mt5 in timeframes.items():
                rates = mt5.copy_rates_from_pos(SYMBOL, tf_mt5, 0, count)
                if rates is None or len(rates) == 0:
                    self.logger.warning(f"未能獲取 {SYMBOL} 的 {tf_str} 數據。"); return None
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df.set_index('time', inplace=True)
                dfs[tf_str] = df
            return dfs
        except Exception as e: self.logger.error(f"獲取市場數據時出錯: {e}"); return None

    def calculate_features(self, market_data: Dict) -> Optional[pd.DataFrame]:
        """實時計算 H1 週期的多時間框架特徵"""
        self.logger.info("正在計算即時特徵...")
        try:
            dataframes = {}
            for tf, df_raw in market_data.items():
                df_with_feats = self.feature_engineer._add_base_features(df_raw)
                if tf == 'D1':
                    df_with_feats['is_uptrend'] = (df_with_feats['close'] > df_with_feats['SMA_50']).astype(int)
                dataframes[tf] = df_with_feats
            final_dataframes = {}
            cols_to_drop = ['open', 'high', 'low', 'close', 'tick_volume']
            if 'D1' in dataframes: final_dataframes['D1'] = dataframes['D1']
            if 'H4' in dataframes and 'D1' in dataframes:
                df_d1_renamed = dataframes['D1'].rename(columns=lambda c: f"D1_{c}" if c not in cols_to_drop else c)
                final_dataframes['H4'] = pd.merge_asof(dataframes['H4'], df_d1_renamed.drop(columns=cols_to_drop, errors='ignore'), left_index=True, right_index=True, direction='backward')
            if 'H1' in dataframes and 'H4' in final_dataframes:
                df_h4_renamed = final_dataframes['H4'].rename(columns=lambda c: f"H4_{c}" if c not in cols_to_drop else c)
                final_dataframes['H1'] = pd.merge_asof(dataframes['H1'], df_h4_renamed.drop(columns=cols_to_drop, errors='ignore'), left_index=True, right_index=True, direction='backward')
            df_h1_final = final_dataframes.get('H1')
            if df_h1_final is None: return None
            df_h1_final.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_h1_final.fillna(method='ffill', inplace=True)
            df_h1_final.dropna(inplace=True)
            self.logger.info("✅ 即時特徵計算完成。")
            return df_h1_final
        except Exception as e:
            self.logger.error(f"計算特徵時發生錯誤: {e}", exc_info=True)
            return None

    def check_for_signal(self, df_features: pd.DataFrame) -> Optional[str]:
        if df_features is None or len(df_features) < 2: return None
        latest_data = df_features.iloc[-2]
        if latest_data[self.features].isnull().any(): self.logger.warning("特徵包含缺失值，跳過預測。"); return None
        features_pred = latest_data[self.features].to_frame().T
        win_prob = self.model.predict_proba(features_pred)[0][1]
        self.logger.info(f"模型預測勝率: {win_prob:.2%}")
        is_uptrend = latest_data.get('H4_D1_is_uptrend', 1.0) > 0.5
        entry_threshold = self.best_params.get('entry_threshold', 0.60)
        if is_uptrend and win_prob > entry_threshold: return "BUY"
        elif not is_uptrend and win_prob < (1 - entry_threshold): return "SELL" 
        return None

    def execute_trade(self, signal: str):
        if mt5.positions_get(symbol=SYMBOL): self.logger.info(f"已持有 {SYMBOL} 倉位，跳過。"); return
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick: self.logger.error("無法獲取最新報價！"); return
        price = tick.ask if signal == "BUY" else tick.bid
        atr_df = pd.DataFrame(mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 20))
        atr_df['tr'] = np.maximum((atr_df['high'] - atr_df['low']), np.maximum(abs(atr_df['high'] - atr_df['close'].shift()), abs(atr_df['low'] - atr_df['close'].shift())))
        atr_val = atr_df['tr'].rolling(14).mean().iloc[-1]
        sl_multiplier = self.best_params.get('sl_atr_multiplier', 1.5)
        tp_multiplier = self.best_params.get('tp_atr_multiplier', 1.5)
        sl_dist, tp_dist = atr_val * sl_multiplier, atr_val * tp_multiplier
        sl = price - sl_dist if signal == "BUY" else price + sl_dist
        tp = price + tp_dist if signal == "BUY" else price - tp_dist
        point = mt5.symbol_info(SYMBOL).point
        lot_size = round((mt5.account_info().balance * RISK_PER_TRADE) / (sl_dist / point * mt5.symbol_info(SYMBOL).trade_tick_size), 2)
        lot_size = max(0.01, lot_size)
        request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": lot_size,
                   "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
                   "price": price, "sl": round(sl, 5), "tp": round(tp, 5), "magic": 20240912,
                   "comment": f"ML_BOT_{TIMEFRAME_STR}", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK}
        self.logger.info(f"準備下單: {request}")
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE: self.logger.error(f"❌ 下單失敗! retcode={result.retcode}")
        else: self.logger.info(f"✅ 下單成功! order ID: {result.order}")

    def run(self):
        if not self.connect_mt5() or not self.load_model_and_params(): return
        self.logger.info(f"🚀 交易機器人啟動，監控 {SYMBOL} on {TIMEFRAME_STR}...")
        while True:
            try:
                now_utc = pd.Timestamp.utcnow()
                next_hour = (now_utc + pd.Timedelta(hours=1)).floor('H')
                wait_seconds = max(1, (next_hour - now_utc).total_seconds() + 5)
                self.logger.info(f"下一根 H1 K線將在 {wait_seconds:.0f} 秒後收盤，進入休眠...")
                time.sleep(wait_seconds)
                self.logger.info("--- 新一輪交易檢查 ---")
                market_data = self.get_market_data()
                if market_data is None: continue
                df_features = self.calculate_features(market_data)
                signal = self.check_for_signal(df_features)
                if signal: self.execute_trade(signal)
            except KeyboardInterrupt: self.logger.info("收到中斷信號..."); break
            except Exception as e: self.logger.error(f"主循環發生嚴重錯誤: {e}", exc_info=True); time.sleep(60)
        mt5.shutdown()
        self.logger.info("MT5 連接已關閉，機器人已停止。")

if __name__ == "__main__":
    logger = setup_logger()
    bot = TradingBot(logger)
    bot.run()
