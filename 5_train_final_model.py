# 檔名: 5_train_final_model.py
# 版本: 1.0
# 描述: 使用所有可用數據和最佳參數，訓練並儲存一個用於實盤交易的最終模型。

import pandas as pd
import lightgbm as lgb
import json
import yaml
from pathlib import Path
import joblib
import logging
import sys

# ==============================================================================
#                      1. 設定區塊
# ==============================================================================
# --- 選擇要為哪個市場訓練最終模型 ---
MARKET_NAME = "EURUSD.sml_H1" # ★★★ 確保這與您在 6_trading_bot_template.py 中設定的 SYMBOL 和 TIMEFRAME_STR 一致 ★★★

# ==============================================================================
#                      2. 主程式邏輯
# ==============================================================================
def setup_logger():
    """設定日誌記錄器"""
    logger = logging.getLogger("FinalModelTrainer")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def create_labels(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """從 4 號腳本移植過來的標籤創建函數"""
    df_out = df.copy()
    tp_m, sl_m, max_hold = settings['tp_atr_multiplier'], settings['sl_atr_multiplier'], settings['max_hold_periods']
    
    atr_col = next((c for c in df_out.columns if 'D1_ATR_14' in c), None) or next((c for c in df_out.columns if 'ATR_14' in c), None)
    if atr_col is None: raise ValueError("數據中缺少 ATR 欄位")
    
    outcomes = pd.Series(index=df_out.index, dtype=float, name='label')
    high_s, low_s, atr_s = df_out['high'], df_out['low'], df_out[atr_col]
    
    for i in range(len(df_out) - max_hold):
        entry, atr = df_out['close'].iloc[i], atr_s.iloc[i]
        if atr <= 0 or pd.isna(atr): continue
        tp, sl = entry + (atr * tp_m), entry - (atr * sl_m)
        highs, lows = high_s.iloc[i+1:i+1+max_hold], low_s.iloc[i+1:i+1+max_hold]
        tp_mask, sl_mask = highs >= tp, lows <= sl
        tp_time = tp_mask.idxmax() if tp_mask.any() else pd.NaT
        sl_time = sl_mask.idxmax() if sl_mask.any() else pd.NaT
        if pd.notna(tp_time) and pd.notna(sl_time): outcomes.iloc[i] = 1 if tp_time <= sl_time else -1
        elif pd.notna(tp_time): outcomes.iloc[i] = 1
        elif pd.notna(sl_time): outcomes.iloc[i] = -1
        else: outcomes.iloc[i] = 0
        
    df_out = df_out.join(outcomes)
    df_out['target_binary'] = (df_out['label'] == 1).astype(int)
    return df_out

def train_and_save_model(logger: logging.Logger):
    """執行完整的模型訓練與儲存流程"""
    logger.info(f"🚀 開始為市場 [{MARKET_NAME}] 訓練最終模型...")
    
    # --- 1. 載入配置和路徑 ---
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        paths = config['paths']
        tb_settings = config['triple_barrier_settings']
        
        market_folder = "_".join(MARKET_NAME.split('_')[:2])
        data_file = Path(paths['features_data']) / market_folder / f"{MARKET_NAME}.parquet"
        features_file = Path(paths['ml_pipeline_output']) / f"selected_features_{MARKET_NAME}.json"
        params_file = Path(paths['ml_pipeline_output']) / f"{MARKET_NAME}_best_params_binary_lgbm.json"
        output_model_file = Path(paths['ml_pipeline_output']) / f"final_model_{MARKET_NAME}.joblib"
        
    except Exception as e:
        logger.error(f"❌ 讀取配置文件或設定路徑時出錯: {e}")
        return

    # --- 2. 載入所需檔案 ---
    try:
        df = pd.read_parquet(data_file)
        with open(features_file, 'r') as f:
            selected_features = json.load(f)['selected_features']
        with open(params_file, 'r') as f:
            # 使用最後一個 Fold 的優化參數作為我們最終模型的超參數
            best_params = json.load(f)['folds_data'][-1]
        logger.info("✅ 數據、特徵列表和最佳參數載入成功。")
    except FileNotFoundError as e:
        logger.error(f"❌ 找不到必要的檔案: {e}")
        logger.error("💡 請確保您已成功運行 2, 3, 4 號腳本。")
        return
        
    # --- 3. 準備最終訓練數據 ---
    logger.info("正在準備最終訓練數據...")
    df_labeled = create_labels(df, tb_settings)
    df_train = df_labeled[df_labeled['label'] != 0].copy()
    df_train.dropna(subset=selected_features + ['target_binary'], inplace=True)
    
    X_train = df_train[selected_features]
    y_train = df_train['target_binary']
    
    if len(X_train) == 0:
        logger.error("❌ 清理後沒有可用的訓練數據！")
        return
    logger.info(f"📊 最終將使用 {len(X_train)} 筆數據進行訓練。")

    # --- 4. 設定模型並訓練 ---
    logger.info("正在設定模型參數並開始訓練...")
    model_params = {k: v for k, v in best_params.items() if k not in ['entry_threshold', 'tp_atr_multiplier', 'sl_atr_multiplier', 'risk_per_trade', 'fold', 'best_sharpe_in_val']}
    model_params.update({'objective': 'binary', 'metric': 'logloss', 'verbosity': -1, 'seed': 42})
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    if pos > 0 and neg > 0: model_params['scale_pos_weight'] = neg / pos
    
    final_model = lgb.LGBMClassifier(**model_params)
    final_model.fit(X_train, y_train)
    logger.info("✅ 最終模型訓練完成！")

    # --- 5. 儲存模型 ---
    try:
        joblib.dump(final_model, output_model_file)
        logger.info(f"💾 模型已成功儲存至: {output_model_file}")
    except Exception as e:
        logger.error(f"❌ 儲存模型時發生錯誤: {e}")

if __name__ == "__main__":
    logger = setup_logger()
    train_and_save_model(logger)
