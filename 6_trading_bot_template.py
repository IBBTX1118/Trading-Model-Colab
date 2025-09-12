# 檔名: 6_trading_bot_template.py
# 版本: 1.1 (H1 策略專用版)
# 描述: 用於連接 OANDA MT5 模擬帳戶並執行 H1 交易策略的機器人模板。

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import sys
from pathlib import Path
import json
import joblib # 使用 joblib 載入 LightGBM 模型

# 假設您的特徵工程腳本位於同一個專案目錄下
from A2_feature_engineering import FeatureEngineer, Config as FeatureConfig

# ==============================================================================
#                      1. 設定區塊
# ==============================================================================
# --- 帳戶設定 (請務必填寫您的真實資訊) ---
MT5_LOGIN = 1600014313                          # ★★★ 請替換為您的 OANDA MT5 模擬帳戶登入號 ★★★
MT5_PASSWORD = "YOUR_PASSWORD"                  # ★★★ 請替換為您的帳戶密碼 ★★★
MT5_SERVER = "OANDA-Demo-1"
MT5_PATH = r"C:\Users\10007793\AppData\Roaming\OANDA MetaTrader 5\terminal64.exe" # ★★★ 請替換為您電腦上 MT5 的完整路徑 ★★★

# --- 策略設定 (已為 H1 預先配置) ---
SYMBOL = "EURUSD.sml" # 要交易的商品 (請確保與您 MT5 中的名稱一致)
TIMEFRAME = mt5.TIMEFRAME_H1
TIMEFRAME_STR = "H1"

# --- 路徑設定 ---
ML_OUTPUT_PATH = Path("Output_ML_Pipeline")
FEATURES_FILE = ML_OUTPUT_PATH / f"selected_features_{SYMBOL}_{TIMEFRAME_STR}.json"
# 注意：這裡我們需要一個最終訓練好的模型，而不是回測過程中的臨時模型
# 建議您在完成所有回測後，運行一個腳本 (例如 5_final_model_training.py)
# 使用所有可用數據訓練一個最終模型並儲存。
FINAL_MODEL_FILE = ML_OUTPUT_PATH / f"final_model_{SYMBOL}_{TIMEFRAME_STR}.joblib" 
BEST_PARAMS_FILE = ML_OUTPUT_PATH / f"{SYMBOL}_{TIMEFRAME_STR}_best_params_binary_lgbm.json"

# --- 交易參數 ---
RISK_PER_TRADE = 0.015 # 每次交易承擔的風險

# ==============================================================================
#                      2. 交易機器人類別
# ==============================================================================
def setup_logger():
    # ... (與之前版本相同)
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
        
        # 實例化特徵工程師，用於即時計算特徵
        feature_config = FeatureConfig()
        self.feature_engineer = FeatureEngineer(feature_config)

    def connect_mt5(self):
        # ... (與之前版本相同)
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
                # 我們假設取最後一個fold的優化參數作為最新參數
                self.best_params = json.load(f)['folds_data'][-1] 
            
            self.logger.info(f"✅ 模型、{len(self.features)} 個特徵及策略參數載入成功。")
            return True
        except Exception as e:
            self.logger.error(f"❌ 載入檔案時發生錯誤: {e}", exc_info=True); return False
    
    def get_market_data(self, count=300): # 需要更多K線來計算指標
        """獲取最新的 H1, H4, D1 數據"""
        try:
            df_h1 = pd.DataFrame(mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, count))
            df_h4 = pd.DataFrame(mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, count))
            df_d1 = pd.DataFrame(mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, count))
            
            for df, tf_str in [(df_h1, "H1"), (df_h4, "H4"), (df_d1, "D1")]:
                if df.empty: self.logger.warning(f"未能獲取 {tf_str} 數據。"); return None
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df.set_index('time', inplace=True)

            return {'H1': df_h1, 'H4': df_h4, 'D1': df_d1}
        except Exception as e: self.logger.error(f"獲取市場數據時出錯: {e}"); return None

    def calculate_features(self, market_data: Dict) -> Optional[pd.DataFrame]:
        """實時計算 H1 週期的多時間框架特徵"""
        self.logger.info("正在計算即時特徵...")
        try:
            # 1. 分別為 D1, H4, H1 計算基礎特徵
            df_d1_feat = self.feature_engineer._add_base_features(market_data['D1'])
            df_d1_feat['is_uptrend'] = (df_d1_feat['close'] > df_d1_feat['SMA_50']).astype(int)
            
            df_h4_feat = self.feature_engineer._add_base_features(market_data['H4'])
            
            df_h1_feat = self.feature_engineer._add_base_features(market_data['H1'])
            
            # 2. 進行多時間框架融合 (與 2_feature_engineering.py 邏輯一致)
            cols_to_drop = ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread']
            
            df_d1_renamed = df_d1_feat.rename(columns=lambda c: f"D1_{c}" if c not in cols_to_drop else c)
            df_h4_merged = pd.merge_asof(df_h4_feat, df_d1_renamed.drop(columns=cols_to_drop, errors='ignore'), left_index=True, right_index=True, direction='backward')

            df_h4_renamed = df_h4_merged.rename(columns=lambda c: f"H4_{c}" if c not in cols_to_drop else c)
            df_h1_final = pd.merge_asof(df_h1_feat, df_h4_renamed.drop(columns=cols_to_drop, errors='ignore'), left_index=True, right_index=True, direction='backward')

            # 3. 對最終的 H1 DataFrame 進行清理
            df_h1_final.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_h1_final.fillna(method='ffill', inplace=True)
            df_h1_final.dropna(inplace=True)

            self.logger.info("✅ 即時特徵計算完成。")
            return df_h1_final
        except Exception as e:
            self.logger.error(f"計算特徵時發生錯誤: {e}", exc_info=True)
            return None

    def check_for_signal(self, df_features: pd.DataFrame):
        """檢查是否有交易信號"""
        if df_features is None or df_features.empty: return None
        
        latest_data = df_features.iloc[-2] # 使用倒數第二根K線 (最新已收盤的)
        if latest_data[self.features].isnull().any(): self.logger.warning("特徵包含缺失值，跳過預測。"); return None

        features_pred = latest_data[self.features].to_frame().T
        win_prob = self.model.predict_proba(features_pred)[0][1]

        self.logger.info(f"模型預測勝率: {win_prob:.2%}")

        is_uptrend = latest_data.get('H4_D1_is_uptrend', 1.0) > 0.5 # 安全地獲取趨勢
        
        entry_threshold = self.best_params.get('entry_threshold', 0.55)

        if is_uptrend and win_prob > entry_threshold: return "BUY"
        elif not is_uptrend and win_prob > entry_threshold: return "SELL" # 假設做空邏輯相同
        
        return None

    def execute_trade(self, signal: str):
        """執行交易下單"""
        if mt5.positions_get(symbol=SYMBOL): self.logger.info(f"已持有 {SYMBOL} 倉位，跳過。"); return
        
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick: self.logger.error("無法獲取最新報價！"); return
        
        price = tick.ask if signal == "BUY" else tick.bid
        atr_val = self.get_latest_data(SYMBOL, TIMEFRAME, 20)['close'].rolling(14).apply(lambda x: np.mean(np.abs(x - x.shift(1)))).iloc[-1] # 簡化ATR計算
        
        sl_multiplier = self.best_params.get('sl_atr_multiplier', 2.0)
        tp_multiplier = self.best_params.get('tp_atr_multiplier', 1.8)
        
        sl_dist = atr_val * sl_multiplier
        tp_dist = atr_val * tp_multiplier

        sl = price - sl_dist if signal == "BUY" else price + sl_dist
        tp = price + tp_dist if signal == "BUY" else price - tp_dist
        
        balance = mt5.account_info().balance
        lot_size = round((balance * RISK_PER_TRADE) / (sl_dist * 100000), 2) # 假設 EURUSD
        lot_size = max(0.01, lot_size)

        request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": lot_size,
                   "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
                   "price": price, "sl": sl, "tp": tp, "magic": 20240912,
                   "comment": f"ML_BOT_{TIMEFRAME_STR}", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}

        self.logger.info(f"準備下單: {request}")
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE: self.logger.error(f"❌ 下單失敗! retcode={result.retcode}")
        else: self.logger.info(f"✅ 下單成功! order ID: {result.order}")

    def run(self):
        """主循環"""
        if not self.connect_mt5() or not self.load_model_and_params(): return
        self.logger.info(f"🚀 交易機器人啟動，監控 {SYMBOL} on {TIMEFRAME_STR}...")
        
        while True:
            try:
                # 計算距離下一根 H1 K線收盤還有多久
                now_utc = pd.Timestamp.utcnow()
                next_hour = (now_utc + pd.Timedelta(hours=1)).floor('H')
                wait_seconds = (next_hour - now_utc).total_seconds() + 5 # 多等5秒確保K線數據更新
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
