# 檔名: 05_model_debugger.ipynb (v2.1 - Colab 顯示修正版)

# 【✅ 解決方案】加入這行魔法指令，讓圖表在 Colab 中顯示
%matplotlib inline

import pandas as pd
import lightgbm as lgb
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# --- 設定區塊 ---
MARKET_NAME = "EURUSD_sml_H4" # 選擇一個市場來進行偵錯
FEATURE_DATA_PATH = Path("Output_Feature_Engineering/MarketData_with_Combined_Features_v3")
ML_OUTPUT_PATH = Path("Output_ML_Pipeline")
CONFIG_PATH = Path("config.yaml")

# --- 1. 載入數據、特徵和設定 ---
print("載入數據中...")
market_folder = MARKET_NAME.split('_')[0] + "_" + MARKET_NAME.split('_')[1]
data_file = FEATURE_DATA_PATH / market_folder / f"{MARKET_NAME}.parquet"
features_file = ML_OUTPUT_PATH / f"selected_features_{MARKET_NAME}.json"

df = pd.read_parquet(data_file)
with open(features_file, 'r') as f:
    features_data = json.load(f)
selected_features = features_data['selected_features']

# 載入 config.yaml 以獲取標籤設定
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
triple_barrier_settings = config['triple_barrier_settings']

# --- 2. 創建標籤並切換至二分類模式 (與 04 腳本完全一致) ---
print("創建標籤並準備數據...")
def create_triple_barrier_labels(df, settings):
    df_out = df.copy()
    tp_multiplier, sl_multiplier, max_hold = settings['tp_atr_multiplier'], settings['sl_atr_multiplier'], settings['max_hold_periods']
    atr_col_name = next((col for col in df_out.columns if 'ATR_14' in col), None)
    if not atr_col_name: raise ValueError("ATR欄位未找到")
    
    outcomes = pd.Series(index=df_out.index, dtype=float, name='label')
    high_series, low_series, atr_series = df_out['high'], df_out['low'], df_out[atr_col_name]

    for i in range(len(df_out) - max_hold):
        entry_price, atr_at_entry = df_out['close'].iloc[i], atr_series.iloc[i]
        if atr_at_entry <= 0 or pd.isna(atr_at_entry): continue
        tp_price, sl_price = entry_price + (atr_at_entry * tp_multiplier), entry_price - (atr_at_entry * sl_multiplier)
        future_highs, future_lows = high_series.iloc[i+1:i+1+max_hold], low_series.iloc[i+1:i+1+max_hold]
        hit_tp_mask, hit_sl_mask = future_highs >= tp_price, future_lows <= sl_price
        tp_hit_time, sl_hit_time = (hit_tp_mask.idxmax() if hit_tp_mask.any() else pd.NaT), (hit_sl_mask.idxmax() if hit_sl_mask.any() else pd.NaT)

        if pd.notna(tp_hit_time) and pd.notna(sl_hit_time): outcomes.iloc[i] = 1 if tp_hit_time <= sl_hit_time else -1
        elif pd.notna(tp_hit_time): outcomes.iloc[i] = 1
        elif pd.notna(sl_hit_time): outcomes.iloc[i] = -1
        else: outcomes.iloc[i] = 0
    return df_out.join(outcomes.to_frame())

df_labeled = create_triple_barrier_labels(df, triple_barrier_settings)
df_trades_only = df_labeled[df_labeled['label'] != 0].copy()
df_trades_only['target_binary'] = (df_trades_only['label'] == 1).astype(int)
df_trades_only.dropna(subset=selected_features + ['target_binary'], inplace=True)
train_size = int(len(df_trades_only) * 0.8)
df_train, df_test = df_trades_only.iloc[:train_size], df_trades_only.iloc[train_size:]
X_train, y_train = df_train[selected_features], df_train['target_binary']
X_test, y_test = df_test[selected_features], df_test['target_binary']

# --- 3. 訓練模型 ---
print("訓練模型...")
neg_count, pos_count = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
model = lgb.LGBMClassifier(objective='binary', scale_pos_weight=scale_pos_weight)
model.fit(X_train, y_train)

# --- 4. 在測試集上進行批量預測 ---
print("在測試集上進行預測...")
win_probabilities = model.predict_proba(X_test)[:, 1]

# --- 5. 視覺化預測概率分佈 ---
print("繪製模型信心分佈圖...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))
sns.histplot(win_probabilities, bins=20, ax=ax, kde=True)
ax.set_title(f'模型預測勝率 (P(Win)) 的分佈 - {MARKET_NAME}', fontsize=16)
ax.set_xlabel('預測的勝率 (Win Probability)', fontsize=12)
ax.set_ylabel('信號數量 (Frequency)', fontsize=12)
ax.axvline(x=0.5, color='r', linestyle='--', label='隨機猜測 (50%)')
ax.legend()
plt.show() # 這行現在會正常工作並顯示圖片

# --- 6. 診斷分析 ---
avg_prob = np.mean(win_probabilities)
confident_buys = np.sum(win_probabilities > 0.6)
confident_sells = np.sum(win_probabilities < 0.4)

print("\n" + "="*50)
print("診斷分析:")
print(f"  - 平均預測勝率: {avg_prob:.2%}")
print(f"  - 高於 60% 信心的買入信號數量: {confident_buys} / {len(win_probabilities)}")
print(f"  - 高於 60% 信心的賣出信號數量: {confident_sells} / {len(win_probabilities)}")
print("="*50)
print("\n如何解讀這張圖:")
print("  - 【理想情況】: 圖形應呈 'U' 型或雙峰型，大量信號集中在 <20% 和 >80% 的區域。")
print("  - 【當前可能的問題】: 圖形可能是一個又高又窄的尖峰，集中在 50% 附近。")
