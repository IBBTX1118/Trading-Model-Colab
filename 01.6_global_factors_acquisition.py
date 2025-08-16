# 檔名: 01.6_global_factors_acquisition.py
# 描述: 從 Investing.com 獲取美元指數(DXY)和VIX指數的歷史數據。
# 版本: 1.0

import investpy
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
    # 定義要獲取的全局因子
    # investpy 需要使用其網站上的確切名稱
    GLOBAL_FACTORS = {
        "DXY": "US Dollar Index",
        "VIX": "CBOE Volatility Index"
    }
    
    # 日期範圍 (與 01.5 保持一致)
    START_DATE = '01/01/2021'
    END_DATE = datetime.now().strftime('%d/%m/%Y')
    
    # 輸出路徑與檔名
    # ★★★ 確保此路徑與您 Google Drive 的專案路徑一致 ★★★
    OUTPUT_PATH = Path("/content/drive/My Drive/Colab_Projects/Trading-Model-Colab/")
    OUTPUT_FILENAME = "global_factors.parquet" # 建議使用 Parquet 格式

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
    all_factors_df = None
    
    try:
        for symbol, name in config.GLOBAL_FACTORS.items():
            logger.info(f"正在獲取 '{name}' ({symbol}) 從 {config.START_DATE} 至 {config.END_DATE} 的日線數據...")
            
            # 獲取指數的日線歷史數據
            df = investpy.get_index_historical_data(
                index=name,
                country='united states', # DXY 和 VIX 都在美國
                from_date=config.START_DATE,
                to_date=config.END_DATE,
                interval='Daily'
            )
            
            # --- 數據整理 ---
            df.index = pd.to_datetime(df.index, utc=True)
            # 只保留收盤價，並重新命名以避免衝突
            df_cleaned = df[['Close']].rename(columns={'Close': f'{symbol}_close'})
            
            # --- 合併數據 ---
            if all_factors_df is None:
                all_factors_df = df_cleaned
            else:
                all_factors_df = all_factors_df.join(df_cleaned, how='outer')
            
            logger.info(f"成功獲取 {len(df)} 筆 {symbol} 的數據。")

        # --- 數據後處理與儲存 ---
        if all_factors_df is not None:
            # 填充缺失值 (例如假日)，使用前一天的值
            all_factors_df.ffill(inplace=True)
            all_factors_df.dropna(inplace=True)
            
            output_file = config.OUTPUT_PATH / config.OUTPUT_FILENAME
            config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
            
            # 儲存為 Parquet 格式，效率更高
            all_factors_df.to_parquet(output_file)
            
            logger.info(f"成功合併 {len(all_factors_df)} 筆全局因子數據，並儲存至：{output_file}")
            logger.info("\n數據預覽：")
            print(all_factors_df.tail())
        else:
            logger.warning("未能獲取任何全局因子數據。")

    except Exception as e:
        logger.error(f"獲取數據時發生錯誤: {e}")
        logger.error("請嘗試更新 investpy 套件: !pip install investpy --upgrade")
        sys.exit(1)

# ==============================================================================
#                                3. 執行區塊
# ==============================================================================
if __name__ == "__main__":
    logger = setup_logger()
    config = Config()
    fetch_and_save_factors(config, logger)
    logger.info("========= 全局因子數據獲取完畢 =========")
