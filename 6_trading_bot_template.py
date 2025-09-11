# 檔名: 6_trading_bot_template.py
# 版本: 1.0
# 描述: 用於連接 OANDA MT5 模擬帳戶並執行交易策略的機器人模板。

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import sys
from pathlib import Path
import joblib # 用於載入/儲存模型
import json

# --- 假設您的特徵工程和模型訓練腳本在同一個項目中 ---
# 這將允許我們直接從其他腳本導入必要的函數
# from feature_engineering_script import FeatureEngineer # 替換為您的特徵工程腳本名
# from training_script import ModelTrainer # 替換為您的模型訓練腳本名

# ==============================================================================
#                      1. 設定區塊
# ==============================================================================
# --- 帳戶設定 (請務必填寫您的真實資訊) ---
# 建議未來將這些敏感資訊移至環境變數或專用的密鑰管理工具
MT5_LOGIN = 12345678      # ★★★ 請替換為您的 OANDA MT5 模擬帳戶登入號 ★★★
MT5_PASSWORD = "YOUR_PASSWORD" # ★★★ 請替換為您的帳戶密碼 ★★★
MT5_SERVER = "OANDA-Demo-1"    # OANDA 的伺服器名稱通常是這個
MT5_PATH = r"C:\Users\YourUser\AppData\Roaming\MetaTrader 5\terminal64.exe" # ★★★ 請替換為您電腦上 MT5 的完整路徑 ★★★

# --- 策略設定 ---
SYMBOL = "EURUSD" # 要交易的商品
TIMEFRAME = mt5.TIMEFRAME_H1 # 要監控的時間週期
TIMEFRAME_STR = "H1"

# --- 模型與特徵路徑 ---
ML_OUTPUT_PATH = Path("Output_ML_Pipeline")
MODEL_FILE = ML_OUTPUT_PATH / f"final_model_{SYMBOL}_{TIMEFRAME_STR}.joblib" # 假設這是最終模型的儲存路徑
FEATURES_FILE = ML_OUTPUT_PATH / f"selected_features_{SYMBOL}_{TIMEFRAME_STR}.json"

# --- 交易參數 ---
RISK_PER_TRADE = 0.01 # 每次交易承擔的風險 (帳戶的1%)
ENTRY_THRESHOLD = 0.60 # 模型預測勝率超過此閾值才進場

# ==============================================================================
#                      2. 輔助函式與類別
# ==============================================================================
def setup_logger():
    """設定日誌記錄器"""
    logger = logging.getLogger("TradingBot")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # 檔案日誌
        fh = logging.FileHandler('trading_bot.log', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # 控制台日誌
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger

class TradingBot:
    def __init__(self, logger):
        self.logger = logger
        self.model = None
        self.features = None
        # 實例化您的特徵工程師 (需要您提供 FeatureEngineer class)
        # self.feature_engineer = FeatureEngineer(...) 

    def connect_mt5(self):
        """連接到 MT5"""
        self.logger.info("正在連接到 MetaTrader 5...")
        if not mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            self.logger.error(f"MT5 初始化失敗, 錯誤代碼 = {mt5.last_error()}")
            return False
        self.logger.info(f"✅ MT5 連接成功! 版本: {mt5.version()}")
        return True

    def load_model_and_features(self):
        """載入訓練好的模型和特徵列表"""
        try:
            self.logger.info(f"正在從 {MODEL_FILE} 載入模型...")
            self.model = joblib.load(MODEL_FILE)
            self.logger.info(f"正在從 {FEATURES_FILE} 載入特徵列表...")
            with open(FEATURES_FILE, 'r') as f:
                self.features = json.load(f)['selected_features']
            self.logger.info(f"✅ 模型與 {len(self.features)} 個特徵載入成功。")
            return True
        except FileNotFoundError as e:
            self.logger.error(f"❌ 錯誤: 找不到模型或特徵檔案: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ 載入模型或特徵時發生未知錯誤: {e}")
            return False
    
    def get_latest_data(self, symbol, timeframe, count=200):
        """獲取最新的K線數據"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                self.logger.warning(f"未能獲取 {symbol} 的數據。")
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('time', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"獲取最新數據時出錯: {e}")
            return None

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算所有必要的特徵"""
        # 這裡需要您調用您的特徵工程邏輯
        # 範例:
        # df_with_features = self.feature_engineer.add_all_features(df)
        # return df_with_features
        self.logger.info("正在計算特徵... (此處為模板，請填入您的特徵計算邏輯)")
        # --- 模板佔位 ---
        # 為了讓模板能運行，我們假設特徵已存在（實際情況需要計算）
        return df 

    def check_for_signal(self, df_latest_features: pd.DataFrame):
        """檢查是否有交易信號"""
        if df_latest_features.empty:
            return None, 0.0

        # 取最後一筆完整的K線數據進行預測
        latest_data = df_latest_features.iloc[-2]
        
        # 檢查是否有缺失值
        if latest_data[self.features].isnull().any():
            self.logger.warning("最新數據的特徵中包含缺失值，跳過預測。")
            return None, 0.0

        features_for_prediction = latest_data[self.features].to_frame().T
        win_probability = self.model.predict_proba(features_for_prediction)[0][1]

        self.logger.info(f"模型預測勝率: {win_probability:.2%}")

        # 獲取趨勢 (需要您的特徵工程邏輯)
        # is_uptrend = latest_data['D1_is_uptrend'] > 0.5
        is_uptrend = True # 模板佔位

        if is_uptrend and win_probability > ENTRY_THRESHOLD:
            return "BUY", win_probability
        elif not is_uptrend and win_probability > (1 - ENTRY_THRESHOLD): # 假設做空邏輯
             return "SELL", win_probability
        
        return None, win_probability

    def execute_trade(self, signal: str, symbol: str):
        """執行交易"""
        # 檢查是否已有持倉
        positions = mt5.positions_get(symbol=symbol)
        if positions and len(positions) > 0:
            self.logger.info(f"已持有 {symbol} 倉位，本次跳過下單。")
            return

        # 獲取價格和ATR用於計算止損止盈
        last_tick = mt5.symbol_info_tick(symbol)
        if not last_tick:
            self.logger.error("無法獲取最新報價，下單失敗。")
            return
        
        price = last_tick.ask if signal == "BUY" else last_tick.bid
        
        # 獲取ATR (您需要從特徵計算中得到這個值)
        # atr_value = ...
        atr_value = 0.0050 # 模板佔位
        
        # 倉位大小計算
        account_info = mt5.account_info()
        balance = account_info.balance
        lot_size = round((balance * RISK_PER_TRADE) / (atr_value * 1.5 * 100000), 2) # 簡化計算
        lot_size = max(0.01, lot_size) # 最小手数

        # 設定止損和止盈
        sl = price - atr_value * 1.5 if signal == "BUY" else price + atr_value * 1.5
        tp = price + atr_value * 1.5 if signal == "BUY" else price - atr_value * 1.5
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": 20240911, # 策略的魔法數字
            "comment": "ML Trading Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        self.logger.info(f"準備下單: {request}")
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"❌ 下單失敗! retcode={result.retcode}, comment={result.comment}")
        else:
            self.logger.info(f"✅ 下單成功! order ID: {result.order}")

    def run(self):
        """主循環"""
        if not self.connect_mt5() or not self.load_model_and_features():
            return
        
        self.logger.info(f"🚀 交易機器人啟動，監控 {SYMBOL} on {TIMEFRAME_STR}...")
        
        while True:
            try:
                # 1. 獲取數據
                df_raw = self.get_latest_data(SYMBOL, TIMEFRAME)
                if df_raw is None:
                    time.sleep(60)
                    continue

                # 2. 計算特徵
                df_features = self.calculate_features(df_raw)

                # 3. 檢查信號
                signal, probability = self.check_for_signal(df_features)

                # 4. 執行交易
                if signal:
                    self.execute_trade(signal, SYMBOL)

                # 等待下一個K線週期
                self.logger.info("等待下一輪檢查...")
                time.sleep(60) # 每分鐘檢查一次

            except KeyboardInterrupt:
                self.logger.info("收到中斷信號，正在關閉機器人...")
                break
            except Exception as e:
                self.logger.error(f"主循環發生嚴重錯誤: {e}", exc_info=True)
                time.sleep(60) # 發生錯誤後等待一段時間再重試

        mt5.shutdown()
        self.logger.info("MT5 連接已關閉，機器人已停止。")


if __name__ == "__main__":
    logger = setup_logger()
    bot = TradingBot(logger)
    bot.run()
