# 檔名: 3_feature_selection_with_diagnostics.py
# 版本: 5.0 (整合診斷版)
# 描述: 在為所有市場進行特徵篩選前，先對一個樣本市場執行快速診斷。

import logging
import sys
import json
import yaml
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import pandas as pd
import numpy as np
import lightgbm as lgb

# ==============================================================================
#                      1. 快速診斷工具 (來自 0_quick_diagnostics.py)
# ==============================================================================
class QuickDiagnostics:
    """快速診斷數據、標籤和基礎策略的健康狀況"""
    def __init__(self, config_path: Path = Path("config.yaml")):
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.tb_settings = config['triple_barrier_settings']
            self.paths = config['paths']
            self.logger.info("✅ (診斷器) 配置檔載入成功")
        except Exception as e:
            self.logger.critical(f"❌ (診斷器) 配置檔載入失敗: {e}")
            raise

    def check_data_files(self, market_name: str):
        self.logger.info("\n🔍 (診斷) 檢查數據檔案...")
        features_dir = Path(self.paths['features_data'])
        market_folder = "_".join(market_name.split('_')[:2])
        market_file = features_dir / market_folder / f"{market_name}.parquet"
        
        if not market_file.exists():
            self.logger.error(f"❌ 特徵數據檔案不存在: {market_file}")
            return False
        self.logger.info(f"✅ 找到目標市場特徵檔案: {market_file}")
        return True

    def check_sample_data(self, market_name: str) -> Optional[pd.DataFrame]:
        self.logger.info(f"\n🔍 (診斷) 檢查 {market_name} 數據品質...")
        features_dir = Path(self.paths['features_data'])
        market_folder = "_".join(market_name.split('_')[:2])
        data_file = features_dir / market_folder / f"{market_name}.parquet"
        try:
            df = pd.read_parquet(data_file)
            self.logger.info(f"✅ 成功載入數據，共 {len(df)} 筆記錄")
            missing_ratio = df.isnull().sum().sum() / df.size
            self.logger.info(f"📊 缺失值比例: {missing_ratio:.2%}")
            if missing_ratio > 0.1: self.logger.warning("⚠️ 缺失值過多")
            return df
        except Exception as e:
            self.logger.error(f"❌ 載入數據失敗: {e}")
            return None

    def test_label_creation(self, df: pd.DataFrame, max_samples=1000) -> Optional[pd.DataFrame]:
        self.logger.info(f"\n🏷️  (診斷) 測試標籤創建...")
        df_test = df.tail(max_samples).copy()
        atr_col = next((c for c in df_test.columns if 'ATR_14' in c), None)
        if atr_col is None: self.logger.error("❌ 找不到 ATR_14 欄位"); return None
        
        try:
            df_labeled = self._create_simple_labels(df_test, atr_col)
            label_counts = df_labeled['label'].value_counts().sort_index()
            total = len(df_labeled.dropna(subset=['label']))
            self.logger.info("📊 標籤分布:")
            for label, count in label_counts.items():
                name = {1: '止盈', -1: '止損', 0: '持有'}[label]
                self.logger.info(f"   {name}: {count} ({count/total:.1%})")
            if total > 0 and (label_counts.min() / total) < 0.1:
                self.logger.warning("⚠️ 標籤可能不平衡")
            return df_labeled
        except Exception as e:
            self.logger.error(f"❌ 標籤創建失敗: {e}")
            return None

    def _create_simple_labels(self, df, atr_col):
        df_out = df.copy()
        tp_m, sl_m, hold = self.tb_settings['tp_atr_multiplier'], self.tb_settings['sl_atr_multiplier'], self.tb_settings['max_hold_periods']
        outcomes = pd.Series(np.nan, index=df_out.index)
        for i in range(len(df_out) - hold):
            entry = df_out['close'].iloc[i]; atr = df_out[atr_col].iloc[i]
            if atr <= 0 or pd.isna(atr): continue
            tp, sl = entry + (atr * tp_m), entry - (atr * sl_m)
            future = df_out.iloc[i+1:i+1+hold]
            hit_tp = (future['high'] >= tp).any(); hit_sl = (future['low'] <= sl).any()
            if hit_tp and hit_sl:
                tp_idx = future[future['high'] >= tp].index[0]
                sl_idx = future[future['low'] <= sl].index[0]
                outcomes.iloc[i] = 1 if tp_idx <= sl_idx else -1
            elif hit_tp: outcomes.iloc[i] = 1
            elif hit_sl: outcomes.iloc[i] = -1
            else: outcomes.iloc[i] = 0
        df_out['label'] = outcomes
        return df_out

    def run_full_diagnosis(self, market_name: str):
        self.logger.info(f"\n{'='*80}\n🚀 開始對樣本市場 [{market_name}] 進行快速診斷...\n{'='*80}")
        if not self.check_data_files(market_name): return
        df = self.check_sample_data(market_name)
        if df is None: return
        self.test_label_creation(df)
        self.logger.info(f"\n{'='*80}\n🚀 快速診斷完畢。\n{'='*80}")


# ==============================================================================
#                      2. 特徵篩選器 (來自 3_feature_selection.py)
# ==============================================================================
class FeatureSelector:
    # ... 此處的 FeatureSelector class 的所有內容與您提供的
    # ... 3_feature_selection.py 完全相同，為了簡潔此處省略。
    # ... 請將您原本的 FeatureSelector class 內容完整複製到這裡。
    def __init__(self, config_path: Path = Path("config.yaml")):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.tb_settings = self.config['triple_barrier_settings']
        self.fs_config = self.config['feature_selection']
        self.paths = self.config['paths']
        self.output_dir = Path(self.paths['ml_pipeline_output'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    def get_feature_importance_for_file(self, df: pd.DataFrame) -> pd.DataFrame:
        non_feature_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'target', 'time', 'spread', 'real_volume', 'label', 'hit_time']
        # 使用自適應標籤
        df_labeled = create_adaptive_labels(df, self.tb_settings)
        features = [col for col in df_labeled.columns if col not in non_feature_cols]
        X, y = df_labeled[features], df_labeled['target']
        combined = pd.concat([X, y], axis=1).dropna()
        X, y = combined[features], combined['target']
        if len(X) == 0 or len(y.value_counts()) < 2: return pd.DataFrame({'feature': [], 'importance': []})
        try:
            model = lgb.LGBMClassifier(**self.fs_config['lgbm_params'])
            model.fit(X, y)
            return pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        except Exception as e: self.logger.error(f"模型訓練失敗: {e}"); return pd.DataFrame({'feature': [], 'importance': []})
    def save_selected_features(self, features: List[str], market_name: str):
        output_path = self.output_dir / f"selected_features_{market_name}.json"
        self.logger.info(f"正在將 {market_name} 的特徵儲存到: {output_path}")
        output_data = {"description": f"為市場 {market_name} 產生的專屬特徵列表", "market": market_name,
                       "feature_count": len(features), "selected_features": features,
                       "generation_settings": {"top_n_features": self.fs_config['top_n_features'], "triple_barrier_settings": self.tb_settings}}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
    def run(self):
        self.logger.info(f"\n{'='*80}\n🚀 開始正式特徵篩選流程...\n{'='*80}")
        input_dir = Path(self.paths['features_data'])
        all_files = list(input_dir.rglob("*.parquet"))
        if not all_files: self.logger.warning("在輸入目錄中沒有找到任何檔案。"); return
        market_files = defaultdict(list)
        for f in all_files: market_files[f.stem].append(f)
        self.logger.info(f"發現 {len(market_files)} 個市場組需要處理: {list(market_files.keys())}")
        for market_name, files in market_files.items():
            self.logger.info(f"\n--- 開始處理市場: {market_name} ---")
            all_importances = []
            for file_path in files:
                try:
                    df = pd.read_parquet(file_path)
                    importance_df = self.get_feature_importance_for_file(df)
                    if not importance_df.empty: all_importances.append(importance_df)
                except Exception as e: self.logger.error(f"處理檔案 {file_path.name} 時出錯: {e}", exc_info=True)
            if not all_importances: self.logger.warning(f"市場 {market_name} 未能計算任何特徵重要性，已跳過。"); continue
            market_importance = pd.concat(all_importances).groupby('feature')['importance'].sum().sort_values(ascending=False)
            top_features = market_importance.head(self.fs_config['top_n_features']).index.tolist()
            self.logger.info(f"✅ 為 {market_name} 選出最重要的 {len(top_features)} 個特徵。")
            self.save_selected_features(top_features, market_name)
        self.logger.info(f"\n{'='*80}\n🚀 特徵篩選流程完畢。\n{'='*80}")

# --- 輔助函式 create_adaptive_labels 和 create_triple_barrier_labels ---
# ... 這兩個函數也需要從您原本的 3_feature_selection.py 複製過來 ...
def create_adaptive_labels(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    df_out = df.copy()
    tp_base, sl_base, max_hold = settings['tp_atr_multiplier'], settings['sl_atr_multiplier'], settings['max_hold_periods']
    adj = {0: 0.8, 1: 0.9, 2: 1.1, 3: 1.2}
    if 'market_regime' in df_out.columns:
        df_out['tp_adj'] = df_out['market_regime'].map(adj) * tp_base
        df_out['sl_adj'] = df_out['market_regime'].map(adj) * sl_base
    else:
        df_out['tp_adj'] = tp_base; df_out['sl_adj'] = sl_base
    atr_col = 'D1_ATR_14' if 'D1_ATR_14' in df_out.columns else 'ATR_14'
    if atr_col not in df_out.columns: raise ValueError(f"缺少 ATR 欄位")
    outcomes = pd.DataFrame(index=df_out.index, columns=['label'])
    high_s, low_s, atr_s = df_out['high'], df_out['low'], df_out[atr_col]
    tp_mult, sl_mult = df_out['tp_adj'], df_out['sl_adj']
    for i in range(len(df_out) - max_hold):
        entry, atr, tp_m, sl_m = df_out['close'].iloc[i], atr_s.iloc[i], tp_mult.iloc[i], sl_mult.iloc[i]
        if atr <= 0 or pd.isna(atr) or pd.isna(tp_m) or pd.isna(sl_m): continue
        tp, sl = entry + (atr * tp_m), entry - (atr * sl_m)
        win_high, win_low = high_s.iloc[i+1:i+1+max_hold], low_s.iloc[i+1:i+1+max_hold]
        tp_mask, sl_mask = win_high >= tp, win_low <= sl
        hit_tp_time = win_high[tp_mask].index.min() if tp_mask.any() else pd.NaT
        hit_sl_time = win_low[sl_mask].index.min() if sl_mask.any() else pd.NaT
        if pd.notna(hit_tp_time) and pd.notna(hit_sl_time): outcomes.loc[df_out.index[i], 'label'] = 1 if hit_tp_time < hit_sl_time else -1
        elif pd.notna(hit_tp_time): outcomes.loc[df_out.index[i], 'label'] = 1
        elif pd.notna(hit_sl_time): outcomes.loc[df_out.index[i], 'label'] = -1
        else: outcomes.loc[df_out.index[i], 'label'] = 0
    df_out = df_out.join(outcomes); df_out['target'] = (df_out['label'] == 1).astype(int)
    df_out.drop(columns=['tp_adj', 'sl_adj'], inplace=True, errors='ignore')
    return df_out
# ==============================================================================
#                      3. 主執行區塊
# ==============================================================================
if __name__ == "__main__":
    # --- 基本設定 ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    CONFIG_PATH = Path("config.yaml")
    DIAGNOSTICS_MARKET_SAMPLE = "EURUSD_sml_H4" # 指定一個用於快速診斷的樣本市場

    try:
        # --- 步驟 1: 執行快速診斷 ---
        diagnostics = QuickDiagnostics(config_path=CONFIG_PATH)
        diagnostics.run_full_diagnosis(market_name=DIAGNOSTICS_MARKET_SAMPLE)
        
        # --- 步驟 2: 執行正式的特徵篩選 ---
        selector = FeatureSelector(config_path=CONFIG_PATH)
        selector.run()

    except Exception as e:
        logging.critical(f"腳本執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
        sys.exit(1)
