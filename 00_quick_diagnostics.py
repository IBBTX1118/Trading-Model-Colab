# 檔名: 00_quick_diagnostics.py
# 描述: 快速診斷模型問題的腳本
# 版本: 1.0

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List

class QuickDiagnostics:
    def __init__(self):
        # 載入配置
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.tb_settings = self.config['triple_barrier_settings']
            print("✅ 配置檔載入成功")
        except Exception as e:
            print(f"❌ 配置檔載入失敗: {e}")
            return

    def check_data_files(self):
        """檢查數據檔案是否存在"""
        print("\n🔍 檢查數據檔案...")
        
        # 檢查特徵數據
        features_dir = Path("Output_Feature_Engineering/MarketData_with_Combined_Features_v3")
        if not features_dir.exists():
            print(f"❌ 特徵數據目錄不存在: {features_dir}")
            return False
            
        feature_files = list(features_dir.rglob("*_H4.parquet"))
        print(f"✅ 找到 {len(feature_files)} 個 H4 特徵檔案")
        
        # 檢查特徵選擇檔案
        ml_dir = Path("Output_ML_Pipeline")
        if not ml_dir.exists():
            print(f"❌ ML管道目錄不存在: {ml_dir}")
            return False
            
        feature_selection_files = list(ml_dir.glob("selected_features_*_H4.json"))
        print(f"✅ 找到 {len(feature_selection_files)} 個特徵選擇檔案")
        
        return len(feature_files) > 0 and len(feature_selection_files) > 0

    def check_sample_data(self, symbol="EURUSD_sml_H4"):
        """檢查單個市場的數據品質"""
        print(f"\n🔍 檢查 {symbol} 數據品質...")
        
        # 載入數據
        data_file = Path(f"Output_Feature_Engineering/MarketData_with_Combined_Features_v3/EURUSD_sml/{symbol}.parquet")
        
        if not data_file.exists():
            print(f"❌ 數據檔案不存在: {data_file}")
            return
            
        try:
            df = pd.read_parquet(data_file)
            print(f"✅ 成功載入數據，共 {len(df)} 筆記錄")
            print(f"📅 時間範圍: {df.index.min()} 至 {df.index.max()}")
        except Exception as e:
            print(f"❌ 載入數據失敗: {e}")
            return
            
        # 檢查基本OHLC數據
        required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
        missing_basic = [col for col in required_cols if col not in df.columns]
        
        if missing_basic:
            print(f"❌ 缺少基本欄位: {missing_basic}")
        else:
            print("✅ 基本 OHLC 欄位完整")
            
        # 檢查ATR欄位
        atr_cols = [col for col in df.columns if 'ATR' in col]
        if atr_cols:
            print(f"✅ 找到 ATR 欄位: {atr_cols[:3]}...")  # 只顯示前3個
        else:
            print("❌ 未找到 ATR 欄位")
            
        # 檢查趨勢欄位
        trend_cols = [col for col in df.columns if 'uptrend' in col.lower()]
        if trend_cols:
            print(f"✅ 找到趨勢欄位: {trend_cols}")
        else:
            print("❌ 未找到趨勢欄位")
        
        # 檢查數據品質
        total_features = len(df.columns)
        missing_ratio = df.isnull().sum().sum() / (len(df) * total_features)
        print(f"📊 特徵總數: {total_features}")
        print(f"📊 缺失值比例: {missing_ratio:.2%}")
        
        if missing_ratio > 0.1:
            print("⚠️  缺失值過多，可能影響模型效果")
            
        return df

    def test_label_creation(self, df, max_samples=1000):
        """測試標籤創建過程"""
        print(f"\n🏷️  測試標籤創建...")
        
        # 取樣測試，避免運算時間過長
        if len(df) > max_samples:
            df_test = df.tail(max_samples).copy()
            print(f"📝 使用最後 {max_samples} 筆數據進行測試")
        else:
            df_test = df.copy()
            
        # 確保有ATR欄位
        atr_col = None
        for col in df_test.columns:
            if 'ATR_14' in col:
                atr_col = col
                break
                
        if atr_col is None:
            print("❌ 找不到 ATR_14 欄位，無法創建標籤")
            return None
            
        print(f"✅ 使用 ATR 欄位: {atr_col}")
        
        # 創建標籤
        try:
            df_labeled = self._create_simple_labels(df_test, atr_col)
            
            # 分析標籤分布
            label_counts = df_labeled['label'].value_counts().sort_index()
            total_labels = len(df_labeled.dropna(subset=['label']))
            
            print(f"📊 標籤分布:")
            for label, count in label_counts.items():
                percentage = count / total_labels * 100 if total_labels > 0 else 0
                label_name = {1: '止盈', -1: '止損', 0: '持有'}[label]
                print(f"   {label_name}: {count} ({percentage:.1f}%)")
                
            # 檢查標籤平衡性
            if total_labels > 0:
                min_ratio = label_counts.min() / total_labels
                if min_ratio < 0.05:
                    print("⚠️  標籤嚴重不平衡！建議調整參數")
                elif min_ratio < 0.15:
                    print("⚠️  標籤輕微不平衡，但可以接受")
                else:
                    print("✅ 標籤分布相對平衡")
                    
            return df_labeled
            
        except Exception as e:
            print(f"❌ 標籤創建失敗: {e}")
            return None

    def _create_simple_labels(self, df, atr_col):
        """簡化的標籤創建函數"""
        df_out = df.copy()
        tp_multiplier = self.tb_settings['tp_atr_multiplier']
        sl_multiplier = self.tb_settings['sl_atr_multiplier']
        max_hold = self.tb_settings['max_hold_periods']
        
        outcomes = []
        
        for i in range(len(df_out) - max_hold):
            if i % 200 == 0:  # 顯示進度
                print(f"   處理進度: {i}/{len(df_out)-max_hold}")
                
            entry_price = df_out['close'].iloc[i]
            atr_at_entry = df_out[atr_col].iloc[i]
            
            if atr_at_entry <= 0 or pd.isna(atr_at_entry):
                outcomes.append(np.nan)
                continue
                
            tp_price = entry_price + (atr_at_entry * tp_multiplier)
            sl_price = entry_price - (atr_at_entry * sl_multiplier)
            
            # 檢查後續價格
            future_data = df_out.iloc[i+1:i+1+max_hold]
            
            if future_data.empty:
                outcomes.append(0)
                continue
                
            hit_tp = (future_data['high'] >= tp_price).any()
            hit_sl = (future_data['low'] <= sl_price).any()
            
            if hit_tp and hit_sl:
                # 檢查哪個先到達
                tp_idx = future_data[future_data['high'] >= tp_price].index[0]
                sl_idx = future_data[future_data['low'] <= sl_price].index[0]
                outcomes.append(1 if tp_idx <= sl_idx else -1)
            elif hit_tp:
                outcomes.append(1)
            elif hit_sl:
                outcomes.append(-1)
            else:
                outcomes.append(0)
        
        # 為剩餘的行填充NaN
        while len(outcomes) < len(df_out):
            outcomes.append(np.nan)
            
        df_out['label'] = outcomes
        return df_out

    def test_simple_backtest(self, df_labeled):
        """簡單的回測測試"""
        print(f"\n📈 執行簡單回測測試...")
        
        if df_labeled is None or 'label' not in df_labeled.columns:
            print("❌ 沒有標籤數據，無法進行回測")
            return
            
        # 模擬簡單的交易
        initial_capital = 100000
        current_capital = initial_capital
        trades = []
        
        # 使用固定風險
        risk_per_trade = 0.02  # 2%
        
        # 找到ATR欄位
        atr_col = None
        for col in df_labeled.columns:
            if 'ATR_14' in col:
                atr_col = col
                break
                
        if atr_col is None:
            print("❌ 找不到 ATR 欄位，無法計算倉位大小")
            return
            
        for i in range(len(df_labeled)):
            if pd.isna(df_labeled['label'].iloc[i]):
                continue
                
            label = df_labeled['label'].iloc[i]
            if label == 0:  # 持有信號，不交易
                continue
                
            entry_price = df_labeled['close'].iloc[i]
            atr_value = df_labeled[atr_col].iloc[i]
            
            if atr_value <= 0 or pd.isna(atr_value):
                continue
                
            # 計算倉位大小（基於風險）
            sl_distance = atr_value * self.tb_settings['sl_atr_multiplier']
            position_size = (current_capital * risk_per_trade) / sl_distance
            
            # 計算盈虧
            tp_distance = atr_value * self.tb_settings['tp_atr_multiplier']
            
            if label == 1:  # 止盈
                pnl = position_size * tp_distance
            else:  # label == -1, 止損
                pnl = -position_size * sl_distance
                
            current_capital += pnl
            trades.append({
                'entry_price': entry_price,
                'label': label,
                'pnl': pnl,
                'capital_after': current_capital
            })
            
            # 限制交易次數，避免過多輸出
            if len(trades) >= 50:
                break
                
        # 分析結果
        if not trades:
            print("❌ 沒有產生任何交易")
            return
            
        total_pnl = current_capital - initial_capital
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) * 100
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        print(f"📊 簡單回測結果:")
        print(f"   總交易次數: {len(trades)}")
        print(f"   總盈虧: {total_pnl:.2f}")
        print(f"   勝率: {win_rate:.1f}%")
        print(f"   平均盈利: {avg_win:.2f}")
        print(f"   平均虧損: {avg_loss:.2f}")
        print(f"   盈虧比: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "   盈虧比: 無限大")
        
        if total_pnl > 0:
            print("✅ 基本策略邏輯似乎有效")
        else:
            print("⚠️  基本策略產生虧損，需要調整參數")

    def run_full_diagnosis(self):
        """執行完整診斷"""
        print("🚀 開始完整診斷...")
        
        # 1. 檢查檔案
        if not self.check_data_files():
            print("❌ 基礎檔案檢查失敗，請確認數據已正確生成")
            return
            
        # 2. 檢查數據品質
        df = self.check_sample_data()
        if df is None:
            print("❌ 數據品質檢查失敗")
            return
            
        # 3. 測試標籤創建
        df_labeled = self.test_label_creation(df)
        if df_labeled is None:
            print("❌ 標籤創建測試失敗")
            return
            
        # 4. 簡單回測
        self.test_simple_backtest(df_labeled)
        
        print("\n🎉 診斷完成！")
        print("\n💡 建議:")
        print("   1. 如果標籤不平衡，請調整 config.yaml 中的 tp_atr_multiplier 和 sl_atr_multiplier")
        print("   2. 如果沒有交易產生，請檢查特徵數據是否包含必要的預測指標")
        print("   3. 如果回測虧損，請考慮調整 entry_threshold 參數")

if __name__ == "__main__":
    diagnostics = QuickDiagnostics()
    diagnostics.run_full_diagnosis()
