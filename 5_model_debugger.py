# 檔名: 5_model_debugger.py (v2.1 - Colab 修復版)
# 描述: 修復在Google Colab中運行的語法錯誤和兼容性問題

import pandas as pd
import lightgbm as lgb
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import warnings

# 設置matplotlib在Colab中正確顯示
import matplotlib
matplotlib.use('Agg')  # 設置後端
plt.ioff()  # 關閉交互模式

# 忽略警告信息
warnings.filterwarnings('ignore')

print("🚀 模型調試工具 - Colab修復版 (v2.1)")
print("="*60)

# --- 設定區塊 ---
MARKET_NAME = "USDJPY_sml_H1"  # 選擇一個市場來進行偵錯
FEATURE_DATA_PATH = Path("Output_Feature_Engineering/MarketData_with_Combined_Features_v3")
ML_OUTPUT_PATH = Path("Output_ML_Pipeline")
CONFIG_PATH = Path("config.yaml")

print(f"📊 目標市場: {MARKET_NAME}")
print(f"🗂️  特徵數據路徑: {FEATURE_DATA_PATH}")
print(f"📁 ML輸出路徑: {ML_OUTPUT_PATH}")

# --- 1. 檢查檔案是否存在 ---
print("\n🔍 檢查檔案...")

market_folder = MARKET_NAME.split('_')[0] + "_" + MARKET_NAME.split('_')[1]
data_file = FEATURE_DATA_PATH / market_folder / f"{MARKET_NAME}.parquet"
features_file = ML_OUTPUT_PATH / f"selected_features_{MARKET_NAME}.json"

if not data_file.exists():
    print(f"❌ 數據檔案不存在: {data_file}")
    print("💡 請確認路徑設定是否正確")
    exit(1)

if not features_file.exists():
    print(f"❌ 特徵檔案不存在: {features_file}")
    print("💡 請先運行 03_feature_selection.py")
    exit(1)

if not CONFIG_PATH.exists():
    print(f"⚠️  配置檔案不存在: {CONFIG_PATH}，使用默認設定")
    triple_barrier_settings = {
        'tp_atr_multiplier': 2.5,
        'sl_atr_multiplier': 1.5,
        'max_hold_periods': 24
    }
else:
    print(f"✅ 找到配置檔案: {CONFIG_PATH}")

# --- 2. 載入數據、特徵和設定 ---
print("\n📂 載入數據中...")
try:
    df = pd.read_parquet(data_file)
    print(f"✅ 數據載入成功: {df.shape}")
except Exception as e:
    print(f"❌ 數據載入失敗: {e}")
    exit(1)

try:
    with open(features_file, 'r', encoding='utf-8') as f:
        features_data = json.load(f)
    selected_features = features_data['selected_features']
    print(f"✅ 特徵載入成功: {len(selected_features)} 個特徵")
except Exception as e:
    print(f"❌ 特徵載入失敗: {e}")
    exit(1)

# 載入config.yaml以獲取標籤設定
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        triple_barrier_settings = config['triple_barrier_settings']
        print(f"✅ 配置載入成功")
    except Exception as e:
        print(f"⚠️  配置載入失敗，使用默認設定: {e}")
        triple_barrier_settings = {
            'tp_atr_multiplier': 2.5,
            'sl_atr_multiplier': 1.5,
            'max_hold_periods': 24
        }

print(f"🎯 標籤設定: TP={triple_barrier_settings['tp_atr_multiplier']}x ATR, SL={triple_barrier_settings['sl_atr_multiplier']}x ATR")

# --- 3. 檢查特徵可用性 ---
print("\n🔍 檢查特徵可用性...")
available_features = [f for f in selected_features if f in df.columns]
missing_features = [f for f in selected_features if f not in df.columns]

if missing_features:
    print(f"⚠️  缺少特徵: {len(missing_features)} 個")
    if len(missing_features) <= 5:
        print(f"   缺少的特徵: {missing_features}")
    else:
        print(f"   前5個缺少的特徵: {missing_features[:5]}")

if len(available_features) < 5:
    print(f"❌ 可用特徵過少 ({len(available_features)})，無法進行調試")
    exit(1)

print(f"✅ 可用特徵: {len(available_features)}/{len(selected_features)}")

# --- 4. 創建標籤並切換至二分類模式 (與 04 腳本完全一致) ---
print("\n🏷️  創建標籤並準備數據...")

def create_triple_barrier_labels(df, settings):
    """創建三道門檻標籤"""
    df_out = df.copy()
    tp_multiplier = settings['tp_atr_multiplier']
    sl_multiplier = settings['sl_atr_multiplier'] 
    max_hold = settings['max_hold_periods']
    
    # 尋找ATR欄位
    atr_col_name = None
    for col in df_out.columns:
        if 'ATR_14' in col:
            atr_col_name = col
            break
    
    if not atr_col_name: 
        raise ValueError("❌ ATR欄位未找到")
    
    print(f"✅ 使用ATR欄位: {atr_col_name}")
    
    outcomes = pd.Series(index=df_out.index, dtype=float, name='label')
    high_series, low_series, atr_series = df_out['high'], df_out['low'], df_out[atr_col_name]

    valid_count = 0
    tp_count = 0
    sl_count = 0
    hold_count = 0

    print("🔄 計算三道門檻標籤...")
    for i in range(len(df_out) - max_hold):
        if i % 5000 == 0:  # 每5000筆顯示進度
            print(f"   進度: {i}/{len(df_out) - max_hold}")
            
        entry_price, atr_at_entry = df_out['close'].iloc[i], atr_series.iloc[i]
        if atr_at_entry <= 0 or pd.isna(atr_at_entry): 
            continue
            
        valid_count += 1
        tp_price = entry_price + (atr_at_entry * tp_multiplier)
        sl_price = entry_price - (atr_at_entry * sl_multiplier)
        
        future_highs = high_series.iloc[i+1:i+1+max_hold]
        future_lows = low_series.iloc[i+1:i+1+max_hold]
        
        hit_tp_mask = future_highs >= tp_price
        hit_sl_mask = future_lows <= sl_price
        
        tp_hit_time = hit_tp_mask.idxmax() if hit_tp_mask.any() else pd.NaT
        sl_hit_time = hit_sl_mask.idxmax() if hit_sl_mask.any() else pd.NaT

        if pd.notna(tp_hit_time) and pd.notna(sl_hit_time): 
            if tp_hit_time <= sl_hit_time:
                outcomes.iloc[i] = 1
                tp_count += 1
            else:
                outcomes.iloc[i] = -1
                sl_count += 1
        elif pd.notna(tp_hit_time): 
            outcomes.iloc[i] = 1
            tp_count += 1
        elif pd.notna(sl_hit_time): 
            outcomes.iloc[i] = -1
            sl_count += 1
        else: 
            outcomes.iloc[i] = 0
            hold_count += 1
    
    print(f"✅ 標籤統計: 有效={valid_count}, 止盈={tp_count}, 止損={sl_count}, 持有={hold_count}")
    return df_out.join(outcomes.to_frame())

try:
    df_labeled = create_triple_barrier_labels(df, triple_barrier_settings)
except Exception as e:
    print(f"❌ 標籤創建失敗: {e}")
    exit(1)

# 只保留有交易信號的數據（排除持有）
df_trades_only = df_labeled[df_labeled['label'] != 0].copy()
df_trades_only['target_binary'] = (df_trades_only['label'] == 1).astype(int)

print(f"📊 交易信號分佈:")
label_counts = df_trades_only['label'].value_counts().sort_index()
print(f"   止盈 (1):  {label_counts.get(1, 0):,} 筆")
print(f"   止損 (-1): {label_counts.get(-1, 0):,} 筆")

# 清理數據
df_trades_only.dropna(subset=available_features + ['target_binary'], inplace=True)

if df_trades_only.empty:
    print("❌ 清理後沒有可用的交易數據")
    exit(1)

print(f"✅ 清理後交易數據: {df_trades_only.shape}")

# 分割訓練和測試集
train_size = int(len(df_trades_only) * 0.8)
df_train = df_trades_only.iloc[:train_size]
df_test = df_trades_only.iloc[train_size:]

X_train = df_train[available_features]
y_train = df_train['target_binary']
X_test = df_test[available_features]
y_test = df_test['target_binary']

print(f"📊 數據分割:")
print(f"   訓練集: {X_train.shape}")
print(f"   測試集: {X_test.shape}")

# --- 5. 訓練模型 ---
print("\n🤖 訓練模型...")

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

print(f"📊 訓練集標籤分佈:")
print(f"   止損 (0): {neg_count:,} 筆")
print(f"   止盈 (1): {pos_count:,} 筆")
print(f"   權重平衡: {scale_pos_weight:.2f}")

try:
    model = lgb.LGBMClassifier(
        objective='binary', 
        scale_pos_weight=scale_pos_weight,
        verbosity=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("✅ 模型訓練成功")
except Exception as e:
    print(f"❌ 模型訓練失敗: {e}")
    exit(1)

# --- 6. 在測試集上進行批量預測 ---
print("\n🔍 在測試集上進行預測...")

try:
    win_probabilities = model.predict_proba(X_test)[:, 1]
    print(f"✅ 預測完成: {len(win_probabilities)} 個預測結果")
except Exception as e:
    print(f"❌ 預測失敗: {e}")
    exit(1)

# --- 7. 視覺化預測概率分佈 ---
print("\n📊 繪製模型信心分佈圖...")

try:
    # 設置圖表樣式 - 使用兼容的樣式
    plt.style.use('default')  # 使用默認樣式而不是可能不存在的seaborn樣式
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 使用matplotlib直接繪製直方圖，避免seaborn版本問題
    ax.hist(win_probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    
    ax.set_title(f'模型預測勝率 (P(Win)) 的分佈 - {MARKET_NAME}', fontsize=16)
    ax.set_xlabel('預測的勝率 (Win Probability)', fontsize=12)
    ax.set_ylabel('信號數量 (Frequency)', fontsize=12)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='隨機猜測 (50%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("✅ 圖表顯示成功")
    
except Exception as e:
    print(f"❌ 圖表顯示失敗: {e}")
    print("💡 這可能是由於環境配置問題，但不影響分析結果")

# --- 8. 診斷分析 ---
print("\n" + "="*60)
print("🔍 診斷分析:")
print("="*60)

avg_prob = np.mean(win_probabilities)
std_prob = np.std(win_probabilities)
min_prob = np.min(win_probabilities)
max_prob = np.max(win_probabilities)

confident_buys = np.sum(win_probabilities > 0.6)
confident_sells = np.sum(win_probabilities < 0.4)
uncertain_signals = np.sum((win_probabilities >= 0.4) & (win_probabilities <= 0.6))

print(f"📊 預測概率統計:")
print(f"   平均預測勝率: {avg_prob:.2%}")
print(f"   標準差: {std_prob:.2%}")
print(f"   最小值: {min_prob:.2%}")
print(f"   最大值: {max_prob:.2%}")

print(f"\n🎯 信號分類:")
print(f"   高信心買入信號 (>60%): {confident_buys} / {len(win_probabilities)} ({confident_buys/len(win_probabilities):.1%})")
print(f"   高信心賣出信號 (<40%): {confident_sells} / {len(win_probabilities)} ({confident_sells/len(win_probabilities):.1%})")
print(f"   不確定信號 (40%-60%): {uncertain_signals} / {len(win_probabilities)} ({uncertain_signals/len(win_probabilities):.1%})")

# --- 9. 概率分布分析 ---
print(f"\n📊 概率區間分佈:")
bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
for i, (low, high) in enumerate(bins):
    count = np.sum((win_probabilities >= low) & (win_probabilities < high))
    percentage = count / len(win_probabilities) * 100
    print(f"   {low*100:2.0f}%-{high*100:2.0f}%: {count:4d} 筆 ({percentage:5.1f}%)")

print("="*60)

# --- 10. 模型診斷建議 ---
print("\n💡 診斷建議:")

if std_prob < 0.05:
    print("⚠️  【問題】: 預測概率標準差過小，模型可能過於保守")
    print("   建議: 檢查特徵工程，增加更多判別性特徵")

if avg_prob > 0.6 or avg_prob < 0.4:
    print("⚠️  【問題】: 平均預測概率偏離50%過多，可能存在偏見")
    print("   建議: 檢查標籤創建邏輯和樣本平衡")

if uncertain_signals / len(win_probabilities) > 0.7:
    print("⚠️  【問題】: 超過70%的信號集中在40%-60%區間，模型信心度不足")
    print("   建議: 增加更多有判別力的特徵，或調整模型參數")

if (confident_buys + confident_sells) / len(win_probabilities) > 0.5:
    print("✅ 【良好】: 超過50%的信號具有較高信心度")

print("\n🎯 理想分佈特徵:")
print("   - 預測概率應呈現雙峰分佈（U型）")
print("   - 大量信號集中在0-20%和80-100%區間") 
print("   - 40-60%區間的信號應盡可能少")
print("   - 標準差應大於10%以保證模型判別能力")

print("\n🎉 調試完成！")
print("="*60)
