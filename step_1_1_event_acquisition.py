# 檔名: 1.1_event_data_acquisition.py
# 描述: 從 Investing.com 獲取宏觀經濟日曆數據，並儲存為 CSV 檔案。
# 版本: 1.0

import investpy
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import logging

# ==============================================================================
#                                1. 設定區塊
# ==============================================================================
class Config:
    """儲存腳本所需的所有設定參數"""
    # 根據您的交易商品，定義相關國家/地區
    COUNTRIES = ['united states', 'euro zone', 'japan', 'united kingdom', 'australia']
    
    # 日期範圍
    START_DATE = '01/01/2021'
    END_DATE = datetime.now().strftime('%d/%m/%Y')
    
    # 只保留中等和高等級別重要性的事件
    IMPORTANCE_FILTER = ['medium', 'high']
    
    # 輸出路徑與檔名
    # ★★★ 請確保此路徑與您 Google Drive 的實際專案路徑一致 ★★★
    OUTPUT_PATH = Path("/content/drive/My Drive/Colab_Projects/Trading-Model-Colab/")
    OUTPUT_FILENAME = "economic_events.csv"

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

def fetch_and_save_events(config: Config, logger: logging.Logger):
    """執行獲取、處理並儲存經濟事件的完整流程"""
    try:
        logger.info(f"正在從 {config.START_DATE} 至 {config.END_DATE} 獲取經濟日曆數據...")
        events_df = investpy.economic_calendar(
            countries=config.COUNTRIES,
            from_date=config.START_DATE,
            to_date=config.END_DATE
        )
        logger.info(f"成功獲取 {len(events_df)} 筆原始事件數據。")

        # --- 數據清洗與整理 ---
        logger.info(f"正在篩選重要性為 {config.IMPORTANCE_FILTER} 的事件...")
        events_df = events_df[events_df['importance'].isin(config.IMPORTANCE_FILTER)].copy()
        
        # 將時間轉換為與我們價格數據相同的 UTC 時區
        # investpy 的時間戳通常是 GMT (等同於 UTC-0)
        events_df['timestamp_utc'] = pd.to_datetime(events_df['date'] + ' ' + events_df['time'], format='%d/%m/%Y %H:%M', errors='coerce').dt.tz_localize('GMT').dt.tz_convert('UTC')
        events_df.dropna(subset=['timestamp_utc'], inplace=True)
        events_df.set_index('timestamp_utc', inplace=True)
        
        # 建立一個數字來表示重要性
        importance_mapping = {'medium': 2, 'high': 3}
        events_df['importance_val'] = events_df['importance'].map(importance_mapping)
        
        # 只保留需要的欄位
        events_df_cleaned = events_df[['event', 'currency', 'importance_val']].sort_index()
        
        # --- 儲存檔案 ---
        output_file = config.OUTPUT_PATH / config.OUTPUT_FILENAME
        # 確保輸出目錄存在
        config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        
        events_df_cleaned.to_csv(output_file)
        
        logger.info(f"成功篩選出 {len(events_df_cleaned)} 筆事件，並儲存至：{output_file}")
        
        logger.info("\n數據預覽：")
        print(events_df_cleaned.head())

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
    fetch_and_save_events(config, logger)
    logger.info("========= 經濟日曆數據獲取完畢 =========")
