# 檔名: 05_model_debugger.ipynb
import pandas as pd
import lightgbm as lgb
import json
from pathlib import Path

# --- 設定區塊 ---
MARKET_NAME = "EURUSD_sml_H4" # 選擇一個市場來進行偵錯
FEATURE_DATA_PATH = Path("Output_Feature_Engineering/MarketData_with_Combined_Features_v3")
ML_OUTPUT_PATH = Path("Output_ML_Pipeline")

# --- 1. 載入數據和特徵 ---
market_folder = MARKET_NAME.split('_')[0] + "_" + MARKET_NAME.split('_')[1]
data_file = FEATURE_DATA_PATH / market_folder / f"{MARKET_NAME}.parquet"
features_file = ML_OUTPUT_PATH / f"selected_features_{MARKET_NAME}.json"
# 注意：您需要先有一個訓練好的模型檔案，這裡假設您在 04 腳本中保存了它
# 如果沒有，可以複製 04 腳本的一小部分來這裡訓練一個臨時模型
# model_file = ML_OUTPUT_PATH / f"{MARKET_NAME}_fold_1_model.lgb" 

df = pd.read_parquet(data_file)
with open(features_file, 'r') as f:
    features_data = json.load(f)
selected_features = features_data['selected_features']

# --- 2. 重新創建標籤和目標映射 (確保與 04 腳本一致) ---
# 這裡直接複製 04 號腳本的標籤創建和映射邏輯
# ... (請將您的 create_adaptive_labels 或 create_triple_barrier_labels 函數貼在這裡)
# ... (以及 triple_barrier_settings 字典)
# df_labeled = create_adaptive_labels(df, triple_barrier_settings)
# mapping = {1: 1, -1: 0, 0: 2} # 止盈:1, 止損:0, 持有:2
# df_labeled['target_multiclass'] = df_labeled['label'].map(mapping)
# df_labeled.dropna(subset=selected_features + ['target_multiclass', 'label'], inplace=True)

# 為了簡化，我們先假設標籤已存在於 04 腳本的輸出中
# 實際使用時，請確保上面的邏輯是完整的
df_labeled = df # 假設 df 已經包含了 'label' 和 'target_multiclass' 欄位

# --- 3. 訓練一個簡單的偵錯模型 ---
train_data = df_labeled.dropna(subset=selected_features + ['target_multiclass'])
X_train = train_data[selected_features]
y_train = train_data['target_multiclass']

print("正在訓練偵錯模型...")
model = lgb.LGBMClassifier(objective='multiclass', num_class=3)
model.fit(X_train, y_train)
print("模型訓練完畢。")

# --- 4. 進行單點預測與驗證 ---
print("\n" + "="*50)
print("單點預測與驗證")
print("="*50)

# 選擇一個測試樣本 (例如，第 1000 筆數據)
sample_index = 1000
X_sample = X_train.iloc[[sample_index]]
y_true_label = train_data['label'].iloc[sample_index]
y_true_class = train_data['target_multiclass'].iloc[sample_index]

# 進行預測
pred_probs = model.predict_proba(X_sample)[0]

# 解析預測結果
# 根據映射關係：class 0=止損, class 1=止盈, class 2=持有
prob_sl = pred_probs[0]
prob_tp = pred_probs[1]
prob_hold = pred_probs[2]

print(f"樣本時間: {X_sample.index[0]}")
print("-" * 50)
print(f"模型預測概率:")
print(f"  - P(止盈 | Win):  {prob_tp:.2%}")
print(f"  - P(止損 | Loss): {prob_sl:.2%}")
print(f"  - P(持有 | Hold): {prob_hold:.2%}")
print("-" * 50)
print(f"數據中的真實標籤 (Ground Truth):")
print(f"  - 真實 Label: {y_true_label} (1=止盈, -1=止損, 0=持有)")
print(f"  - 真實 Class: {y_true_class}")
print("="*50)

# --- 5. 邏輯診斷 ---
print("診斷結論:")
predicted_class = pred_probs.argmax()
if predicted_class == 1 and y_true_label == 1:
    print("✅ 邏輯正確：模型預測'止盈'，實際也為'止盈'。")
elif predicted_class == 0 and y_true_label == -1:
    print("✅ 邏輯正確：模型預測'止損'，實際也為'止損'。")
elif predicted_class == 1 and y_true_label == -1:
    print("❌ 邏輯反轉：模型預測'止盈'，但實際為'止損'！這是問題的根源！")
elif predicted_class == 0 and y_true_label == 1:
    print("❌ 邏輯反轉：模型預測'止損'，但實際為'止盈'！這是問題的根源！")
else:
    print("ℹ️  模型預測與實際標籤不符，但非直接反轉 (例如預測持有但實際止盈/損)。")
