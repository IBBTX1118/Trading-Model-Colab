# 檔名: 05_model_debugger.py
# 描述: 模型調試工具，用於診斷預測邏輯和檢查數據品質
# 版本: 1.1 (修復版 - 包含完整標籤創建邏輯)

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import yaml
from pathlib import Path
import sys

# ==============================================================================
#                      標籤創建函數 (與04腳本保持一致)
# ==============================================================================
def create_adaptive_labels(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """自適應標籤創建，根據市場狀態動態調整止盈止損"""
    df_out = df.copy()
    
    # 基礎倍數
    tp_multiplier_base = settings['tp_atr_multiplier']
    sl_multiplier_base = settings['sl_atr_multiplier']
    max_hold = settings['max_hold_periods']
    
    # 狀態調整因子
    regime_adjustment = {
        0: 0.8,  # 低波動盤整: 收緊目標
        1: 0.9,  # 低波動趨勢: 略微收緊
        2: 1.1,  # 高波動盤整: 略微放大
        3: 1.2   # 高波動趨勢: 放大目標
    }
    
    # 檢查 market_regime 是否存在
    if 'market_regime' in df_out.columns:
        df_out['tp_multiplier_adj'] = df_out['market_regime'].map(regime_adjustment) * tp_multiplier_base
        df_out['sl_multiplier_adj'] = df_out['market_regime'].map(regime_adjustment) * sl_multiplier_base
        print("✅ 使用市場狀態自適應調整止盈止損倍數")
    else:
        # 如果沒有狀態特徵，則退回使用固定倍數
        df_out['tp_multiplier_adj'] = tp_multiplier_base
        df_out['sl_multiplier_adj'] = sl_multiplier_base
        print("⚠️  未找到 market_regime 特徵，使用固定止盈止損倍數")
    
    # 檢查ATR欄位
    atr_col_name = None
    for col in df_out.columns:
        if 'D1_ATR_14' in col:
            atr_col_name = col
            break
        elif 'ATR_14' in col:
            atr_col_name = col
            
    if atr_col_name is None:
        raise ValueError(f"❌ 數據中缺少 ATR 欄位，無法創建標籤。")
    
    print(f"✅ 使用ATR欄位: {atr_col_name}")
    
    # 創建標籤結果
    outcomes = pd.Series(index=df_out.index, dtype=float, name='label')
    
    high_series, low_series, atr_series = df_out['high'], df_out['low'], df_out[atr_col_name]
    tp_multipliers, sl_multipliers = df_out['tp_multiplier_adj'], df_out['sl_multiplier_adj']
    
    valid_count = 0
    tp_count = 0
    sl_count = 0
    hold_count = 0
    
    print("🔄 開始計算三道門檻標籤...")
    
    # 為每個時間點計算三道門檻標籤
    for i in range(len(df_out) - max_hold):
        if i % 1000 == 0:  # 每1000筆顯示進度
            print(f"   處理進度: {i}/{len(df_out) - max_hold}")
            
        entry_price = df_out['close'].iloc[i]
        atr_at_entry = atr_series.iloc[i]
        tp_multiplier = tp_multipliers.iloc[i]
        sl_multiplier = sl_multipliers.iloc[i]
        
        # 檢查數據有效性
        if atr_at_entry <= 0 or pd.isna(atr_at_entry) or pd.isna(tp_multiplier) or pd.isna(sl_multiplier):
            continue
            
        valid_count += 1
        
        # 計算自適應的止盈止損價格
        tp_price = entry_price + (atr_at_entry * tp_multiplier)
        sl_price = entry_price - (atr_at_entry * sl_multiplier)
        
        # 檢查未來價格行為
        future_highs = high_series.iloc[i+1:i+1+max_hold]
        future_lows = low_series.iloc[i+1:i+1+max_hold]
        
        if future_highs.empty or future_lows.empty:
            continue
            
        # 檢查觸及條件
        hit_tp_mask = future_highs >= tp_price
        hit_sl_mask = future_lows <= sl_price
        
        tp_hit = hit_tp_mask.any()
        sl_hit = hit_sl_mask.any()
        
        if tp_hit and sl_hit:
            # 都觸及，看誰先
            tp_first_idx = hit_tp_mask.idxmax() if tp_hit else None
            sl_first_idx = hit_sl_mask.idxmax() if sl_hit else None
            
            if tp_first_idx <= sl_first_idx:
                outcomes.iloc[i] = 1
                tp_count += 1
            else:
                outcomes.iloc[i] = -1
                sl_count += 1
        elif tp_hit:
            outcomes.iloc[i] = 1
            tp_count += 1
        elif sl_hit:
            outcomes.iloc[i] = -1
            sl_count += 1
        else:
            outcomes.iloc[i] = 0
            hold_count += 1
    
    print(f"✅ 自適應標籤統計: 有效={valid_count}, 止盈={tp_count}, 止損={sl_count}, 持有={hold_count}")
    
    # 合併結果並創建目標變數
    df_out = df_out.join(outcomes.to_frame())
    df_out['target'] = (df_out['label'] == 1).astype(int)
    
    # 清理臨時欄位
    df_out.drop(columns=['tp_multiplier_adj', 'sl_multiplier_adj'], inplace=True, errors='ignore')
    
    return df_out

def load_config(config_path: str = 'config.yaml') -> dict:
    """載入配置檔案"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️  警告: 找不到配置檔案 {config_path}，使用默認設置")
        return {
            'triple_barrier_settings': {
                'tp_atr_multiplier': 2.5,
                'sl_atr_multiplier': 1.5,
                'max_hold_periods': 24
            }
        }
    except Exception as e:
        print(f"❌ 讀取配置檔案失敗: {e}")
        return {}

# ==============================================================================
#                      主程序
# ==============================================================================
def main():
    print("🚀 模型調試工具啟動 (版本 1.1)")
    print("="*60)
    
    # --- 設定區塊 ---
    MARKET_NAME = "EURUSD_sml_H4"  # 可以修改為其他市場
    FEATURE_DATA_PATH = Path("Output_Feature_Engineering/MarketData_with_Combined_Features_v3")
    ML_OUTPUT_PATH = Path("Output_ML_Pipeline")
    USE_ADAPTIVE_LABELS = True  # 是否使用自適應標籤
    
    print(f"📊 目標市場: {MARKET_NAME}")
    print(f"🗂️  特徵數據路徑: {FEATURE_DATA_PATH}")
    print(f"📁 ML輸出路徑: {ML_OUTPUT_PATH}")
    print(f"🎯 標籤方法: {'自適應標籤' if USE_ADAPTIVE_LABELS else '固定倍數標籤'}")
    print("-"*60)
    
    # --- 1. 載入配置 ---
    print("🔧 載入配置檔案...")
    config = load_config()
    tb_settings = config.get('triple_barrier_settings', {
        'tp_atr_multiplier': 2.5,
        'sl_atr_multiplier': 1.5,
        'max_hold_periods': 24
    })
    print(f"✅ 三道門檻設定: TP={tb_settings['tp_atr_multiplier']}x ATR, SL={tb_settings['sl_atr_multiplier']}x ATR, 持有={tb_settings['max_hold_periods']}期")
    
    # --- 2. 載入數據和特徵 ---
    print("\n📂 載入數據和特徵檔案...")
    
    # 構建檔案路徑
    market_folder = MARKET_NAME.split('_')[0] + "_" + MARKET_NAME.split('_')[1]
    data_file = FEATURE_DATA_PATH / market_folder / f"{MARKET_NAME}.parquet"
    features_file = ML_OUTPUT_PATH / f"selected_features_{MARKET_NAME}.json"
    
    # 檢查檔案是否存在
    if not data_file.exists():
        print(f"❌ 數據檔案不存在: {data_file}")
        return
        
    if not features_file.exists():
        print(f"❌ 特徵檔案不存在: {features_file}")
        print("💡 提示: 請先運行 03_feature_selection.py 生成特徵檔案")
        return
    
    # 載入數據
    print(f"📊 載入數據檔案: {data_file.name}")
    df = pd.read_parquet(data_file)
    print(f"   數據形狀: {df.shape}")
    print(f"   時間範圍: {df.index.min()} 到 {df.index.max()}")
    
    # 載入特徵
    print(f"🎯 載入特徵檔案: {features_file.name}")
    with open(features_file, 'r', encoding='utf-8') as f:
        features_data = json.load(f)
    selected_features = features_data['selected_features']
    print(f"   選中特徵數量: {len(selected_features)}")
    
    # 檢查特徵可用性
    available_features = [f for f in selected_features if f in df.columns]
    missing_features = [f for f in selected_features if f not in df.columns]
    
    if missing_features:
        print(f"⚠️  缺少特徵: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
        print(f"   (共 {len(missing_features)} 個缺少特徵)")
    
    if len(available_features) < 5:
        print(f"❌ 可用特徵過少 ({len(available_features)})，無法進行調試")
        return
    
    print(f"✅ 使用 {len(available_features)}/{len(selected_features)} 個特徵")
    
    # --- 3. 創建標籤 ---
    print("\n🏷️  創建標籤...")
    if USE_ADAPTIVE_LABELS:
        df_labeled = create_adaptive_labels(df, tb_settings)
    else:
        print("⚠️  固定倍數標籤創建邏輯未實現，請使用自適應標籤")
        return
    
    # 創建多分類目標變數
    mapping = {1: 1, -1: 0, 0: 2}  # 止盈:1, 止損:0, 持有:2
    df_labeled['target_multiclass'] = df_labeled['label'].map(mapping)
    
    # 檢查標籤分佈
    label_counts = df_labeled['label'].value_counts().sort_index()
    class_counts = df_labeled['target_multiclass'].value_counts().sort_index()
    
    print(f"📊 標籤分佈:")
    print(f"   止盈 (1):  {label_counts.get(1, 0):,} 筆")
    print(f"   止損 (-1): {label_counts.get(-1, 0):,} 筆")
    print(f"   持有 (0):  {label_counts.get(0, 0):,} 筆")
    print(f"📊 分類標籤分佈:")
    print(f"   Class 0 (止損): {class_counts.get(0, 0):,} 筆")
    print(f"   Class 1 (止盈): {class_counts.get(1, 0):,} 筆")
    print(f"   Class 2 (持有): {class_counts.get(2, 0):,} 筆")
    
    # --- 4. 準備訓練數據 ---
    print("\n🔧 準備訓練數據...")
    train_data = df_labeled.dropna(subset=available_features + ['target_multiclass', 'label'])
    
    if train_data.empty:
        print("❌ 清理後數據為空，無法進行調試")
        return
        
    print(f"✅ 清理後數據形狀: {train_data.shape}")
    
    X_train = train_data[available_features]
    y_train = train_data['target_multiclass']
    
    # --- 5. 訓練調試模型 ---
    print("\n🤖 訓練調試模型...")
    model_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'seed': 42,
        'n_estimators': 100  # 較少的樹，加快調試速度
    }
    
    model = lgb.LGBMClassifier(**model_params)
    
    try:
        model.fit(X_train, y_train)
        print("✅ 模型訓練完畢")
    except Exception as e:
        print(f"❌ 模型訓練失敗: {e}")
        return
    
    # --- 6. 進行單點預測與驗證 ---
    print("\n" + "="*60)
    print("🔍 單點預測與驗證")
    print("="*60)
    
    # 選擇測試樣本
    if len(X_train) < 1000:
        sample_index = len(X_train) // 2
    else:
        sample_index = 1000
    
    X_sample = X_train.iloc[[sample_index]]
    y_true_label = train_data['label'].iloc[sample_index]
    y_true_class = train_data['target_multiclass'].iloc[sample_index]
    
    # 進行預測
    try:
        pred_probs = model.predict_proba(X_sample)[0]
    except Exception as e:
        print(f"❌ 預測失敗: {e}")
        return
    
    # 解析預測結果
    prob_sl = pred_probs[0]    # Class 0: 止損
    prob_tp = pred_probs[1]    # Class 1: 止盈  
    prob_hold = pred_probs[2]  # Class 2: 持有
    
    print(f"🕐 樣本時間: {X_sample.index[0]}")
    print(f"📍 樣本索引: {sample_index}")
    print("-" * 60)
    print(f"🤖 模型預測概率:")
    print(f"   📈 P(止盈):  {prob_tp:.2%}")
    print(f"   📉 P(止損):  {prob_sl:.2%}")
    print(f"   ⏸️  P(持有):  {prob_hold:.2%}")
    
    predicted_class = pred_probs.argmax()
    confidence = pred_probs.max() - np.sort(pred_probs)[-2]
    
    print("-" * 60)
    print(f"📊 預測結果:")
    print(f"   🎯 預測類別: {predicted_class} ({'止損' if predicted_class == 0 else '止盈' if predicted_class == 1 else '持有'})")
    print(f"   🎪 預測信心度: {confidence:.2%}")
    
    print("-" * 60)
    print(f"📋 真實標籤 (Ground Truth):")
    print(f"   🏷️  真實 Label: {y_true_label} ({'止盈' if y_true_label == 1 else '止損' if y_true_label == -1 else '持有'})")
    print(f"   🏷️  真實 Class: {y_true_class}")
    print("="*60)
    
    # --- 7. 邏輯診斷 ---
    print("🔍 邏輯診斷:")
    
    diagnosis_correct = False
    if predicted_class == 1 and y_true_label == 1:
        print("✅ 邏輯正確：模型預測'止盈'，實際也為'止盈'。")
        diagnosis_correct = True
    elif predicted_class == 0 and y_true_label == -1:
        print("✅ 邏輯正確：模型預測'止損'，實際也為'止損'。")
        diagnosis_correct = True
    elif predicted_class == 2 and y_true_label == 0:
        print("✅ 邏輯正確：模型預測'持有'，實際也為'持有'。")
        diagnosis_correct = True
    elif predicted_class == 1 and y_true_label == -1:
        print("❌ 邏輯反轉：模型預測'止盈'，但實際為'止損'！這是問題的根源！")
    elif predicted_class == 0 and y_true_label == 1:
        print("❌ 邏輯反轉：模型預測'止損'，但實際為'止盈'！這是問題的根源！")
    else:
        print("ℹ️  模型預測與實際標籤不符，但非直接反轉 (例如預測持有但實際止盈/損)。")
    
    print(f"🎯 單樣本準確性: {'正確' if diagnosis_correct else '錯誤'}")
    
    # --- 8. 批量驗證 ---
    print("\n" + "="*60)
    print("📊 批量驗證 (隨機選取100個樣本)")
    print("="*60)
    
    # 隨機選取樣本進行批量測試
    test_indices = np.random.choice(len(X_train), min(100, len(X_train)), replace=False)
    
    correct_predictions = 0
    logic_reversals = 0
    
    for idx in test_indices:
        X_test = X_train.iloc[[idx]]
        y_true_label_test = train_data['label'].iloc[idx]
        
        pred_probs_test = model.predict_proba(X_test)[0]
        predicted_class_test = pred_probs_test.argmax()
        
        # 檢查預測正確性
        if (predicted_class_test == 1 and y_true_label_test == 1) or \
           (predicted_class_test == 0 and y_true_label_test == -1) or \
           (predicted_class_test == 2 and y_true_label_test == 0):
            correct_predictions += 1
        
        # 檢查邏輯反轉
        if (predicted_class_test == 1 and y_true_label_test == -1) or \
           (predicted_class_test == 0 and y_true_label_test == 1):
            logic_reversals += 1
    
    accuracy = correct_predictions / len(test_indices)
    reversal_rate = logic_reversals / len(test_indices)
    
    print(f"📈 批量測試結果:")
    print(f"   🎯 準確率: {accuracy:.2%} ({correct_predictions}/{len(test_indices)})")
    print(f"   🔄 邏輯反轉率: {reversal_rate:.2%} ({logic_reversals}/{len(test_indices)})")
    
    if reversal_rate > 0.1:  # 如果反轉率超過10%
        print("⚠️  警告: 邏輯反轉率較高，可能存在標籤創建或映射問題！")
    elif accuracy > 0.6:
        print("✅ 模型邏輯基本正常")
    else:
        print("ℹ️  準確率較低，可能需要更多特徵工程或參數調整")
    
    # --- 9. 特徵重要性分析 ---
    print("\n" + "="*60)
    print("📊 特徵重要性分析 (Top 10)")
    print("="*60)
    
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False, float_format="%.4f"))
    
    print("\n" + "="*60)
    print("🎉 調試完成！")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  用戶中斷執行")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程式執行時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
