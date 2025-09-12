# 檔名: 1.2_global_factors_acquisition.py
# 描述: 從 Yahoo Finance 獲取美元指數(DXY)和VIX指數的歷史數據。
# 版本: 2.0 (改用 yfinance 提高穩定性)

import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import sys

# ==============================================================================
#                                1. 設定區塊
# ==============================================================================
class Config:
    """儲存腳本所需的所有設定參數"""
    # Yahoo Finance 使用的 Ticker
    GLOBAL_FACTORS = {
        "DXY": "DX-Y.NYB",
        "VIX": "^VIX"
    }
    
    # 日期範圍
    START_DATE = '2021-01-01'
    # yfinance 會自動抓取到最新的可用數據，無需手動設定 END_DATE
    
    # 輸出路徑與檔名
    # ★★★ 確保此路徑與您 Google Drive 的專案路徑一致 ★★★
    OUTPUT_PATH = Path("/content/drive/My Drive/Colab_Projects/Trading-Model-Colab/")
    OUTPUT_FILENAME = "global_factors.parquet"

# ==============================================================================
#                                2. 主程式邏輯
# ==============================================================================
def setup_logger() -> logging.Logger:
    """設定日誌記錄器"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def fetch_and_save_factors(config: Config, logger: logging.Logger):
    """獲取、合併並儲存所有全局因子數據"""
    all_factors_list = []
    
    try:
        tickers = list(config.GLOBAL_FACTORS.values())
        
        logger.info(f"正在從 Yahoo Finance 獲取 {tickers} 從 {config.START_DATE} 開始的日線數據...")
        
        # yfinance 可以一次性下載多個 tickers 的數據
        data = yf.download(tickers, start=config.START_DATE)
        
        # --- 數據整理 ---
        # 只保留收盤價
        df_close = data['Close']
        
        # 遍歷設定的因子，處理並重新命名欄位
        for symbol, ticker in config.GLOBAL_FACTORS.items():
            if ticker in df_close.columns:
                # 重新命名欄位以符合後續流程 (例如 DX-Y.NYB -> DXY_close)
                factor_df = df_close[[ticker]].rename(columns={ticker: f'{symbol}_close'})
                all_factors_list.append(factor_df)
                logger.info(f"成功處理 {len(factor_df.dropna())} 筆 {symbol} 的數據。")

        # --- 合併數據 ---
        if not all_factors_list:
            logger.error("未能獲取任何全局因子數據。")
            return
            
        all_factors_df = pd.concat(all_factors_list, axis=1)
        
        # --- 數據後處理與儲存 ---
        # 確保索引是 UTC 時區，以便與您的價格數據對齊
        all_factors_df.index = all_factors_df.index.tz_localize('UTC')
        
        # 填充缺失值 (例如假日)，使用前一天的值
        all_factors_df.ffill(inplace=True)
        all_factors_df.dropna(inplace=True)
        
        output_file = config.OUTPUT_PATH / config.OUTPUT_FILENAME
        config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        
        # 儲存為 Parquet 格式
        all_factors_df.to_parquet(output_file)
        
        logger.info(f"成功合併 {len(all_factors_df)} 筆全局因子數據，並儲存至：{output_file}")
        logger.info("\n數據預覽：")
        print(all_factors_df.tail())

    except Exception as e:
        logger.error(f"獲取數據時發生錯誤: {e}", exc_info=True)
        sys.exit(1)

# ==============================================================================
#                                3. 執行區塊
# ==============================================================================
if __name__ == "__main__":
    logger = setup_logger()
    config = Config()
    fetch_and_save_factors(config, logger)
    logger.info("========= 全局因子數據獲取完畢 (yfinance) =========")
