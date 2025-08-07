# 檔名: 04_parameter_optimization.py
# 描述: Phase 2 實作：整合三道門檻標籤法與夏普比率優化。
# 版本: 6.0 (Phase 2 Alpha)

import sys
import yaml
import json
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from datetime import timedelta
import traceback

import backtrader as bt
import lightgbm as lgb
import optuna

# ==============================================================================
# 1. 載入設定與輔助函式 (與之前相同)
# ==============================================================================
def load_config(config_path: str = 'config.yaml') -> Dict:
    # ... (此函式內容不變，為節省空間省略)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"致命錯誤: 設定檔 {config_path} 不存在！")
        sys.exit(1)
    return {} # 加上一個空的返回值以防萬一

# ==============================================================================
# 2. ★★★ 新增：三道門檻標籤法實作 ★★★
# ==============================================================================
def create_triple_barrier_labels(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """為數據集創建三道門檻標籤"""
    df_out = df.copy()
    tp_pct = settings['tp_pct']
    sl_pct = settings['sl_pct']
    max_hold = settings['max_hold_periods']
    
    outcomes = pd.DataFrame(index=df_out.index, columns=['hit_time', 'label'])
    
    for i in range(len(df_out) - max_hold):
        entry_price = df_out['close'].iloc[i]
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
        
        window = df_out.iloc[i+1 : i+1+max_hold]
        
        # 尋找觸及停利和停損的時間點
        hit_tp_time = window[window['high'] >= tp_price].index.min()
        hit_sl_time = window[window['low'] <= sl_price].index.min()
        
        if pd.notna(hit_tp_time) and pd.notna(hit_sl_time):
            # 如果兩者都觸及，取較早發生者
            if hit_tp_time < hit_sl_time:
                outcomes.loc[df_out.index[i], 'label'] = 1  # 停利
                outcomes.loc[df_out.index[i], 'hit_time'] = hit_tp_time
            else:
                outcomes.loc[df_out.index[i], 'label'] = -1 # 停損
                outcomes.loc[df_out.index[i], 'hit_time'] = hit_sl_time
        elif pd.notna(hit_tp_time):
            outcomes.loc[df_out.index[i], 'label'] = 1  # 停利
            outcomes.loc[df_out.index[i], 'hit_time'] = hit_tp_time
        elif pd.notna(hit_sl_time):
            outcomes.loc[df_out.index[i], 'label'] = -1 # 停損
            outcomes.loc[df_out.index[i], 'hit_time'] = hit_sl_time
        else:
            outcomes.loc[df_out.index[i], 'label'] = 0  # 時間到期
            outcomes.loc[df_out.index[i], 'hit_time'] = window.index[-1]
            
    df_out = df_out.join(outcomes)
    
    # 為了簡化為二元分類問題，我們只預測是否會觸及停利
    df_out['target'] = (df_out['label'] == 1).astype(int)
    return df_out

# ==============================================================================
# 3. Backtrader 策略 (與之前相同)
# ==============================================================================
class FinalMLStrategy(bt.Strategy):
    # ... (此類別內容不變，為節省空間省略)
    params = (('model', None),('features', None),('entry_threshold', 0.55),('atr_period', 14),('atr_sl_multiplier', 2.0),('atr_tp_multiplier', 3.0),)
    def __init__(self):
        if not self.p.model or not self.p.features: raise ValueError("模型和特徵列表必須提供！")
        self.feature_lines = [getattr(self.data.lines, f) for f in self.p.features]
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
    def next(self):
        if self.position: return
        try: feature_vector = np.array([line[0] for line in self.feature_lines]).reshape(1, -1)
        except IndexError: return
        pred_prob = self.p.model.predict_proba(feature_vector)[0][1]
        if pred_prob > self.p.entry_threshold:
            current_price = self.data.close[0]; current_atr = self.atr[0]
            if current_atr > 0:
                sl_price = current_price - self.p.atr_sl_multiplier * current_atr
                tp_price = current_price + self.p.atr_tp_multiplier * current_atr
                self.buy_bracket(price=current_price, stopprice=sl_price, limitprice=tp_price,)

# ==============================================================================
# 4. 主流程控制器 (★★★ 重大修改 ★★★)
# ==============================================================================
class MLOptimizerAndBacktester:
    def __init__(self, config: Dict):
        self.config = config
        self.paths = config['paths']
        self.wfo_config = config['walk_forward_optimization']
        self.strategy_params = config['strategy_params']
        self.tb_settings = config['triple_barrier_settings']
        
        self.output_base_dir = Path(self.paths['ml_pipeline_output'])
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        features_file = self.output_base_dir / self.paths['selected_features_filename']
        self.selected_features = self._load_json(features_file)['selected_features']
        
        self.all_market_results = {}
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _load_json(self, file_path: Path) -> Dict:
        # ... (此函式內容不變，為節省空間省略)
        if not file_path.exists():
            print(f"致命錯誤: 輸入檔案 {file_path} 不存在！")
            sys.exit(1)
        with open(file_path, 'r') as f: return json.load(f)
        return {}
    
    # ★★★ Optuna 目標函式：重大修改，改為優化夏普比率 ★★★
    def objective(self, trial: optuna.trial.Trial, X_train, y_train, df_val) -> float:
        """Optuna 的目標函數，優化驗證集上的夏普比率。"""
        param = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'seed': 42, 'n_jobs': -1,
        }
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train.values, y_train.values)
        
        # 在驗證集上執行迷你回測
        result = self.run_backtest_on_fold(df_val, model)
        
        # 從回測結果中提取夏普比率
        sharpe = result.get('sharpe_ratio', -1.0) # 如果沒有交易，給予負分
        
        return sharpe if sharpe is not None else -1.0

    def run_backtest_on_fold(self, df_fold: pd.DataFrame, model: lgb.LGBMClassifier) -> Dict:
        # ... (此函式內容不變，為節省空間省略)
        class PandasDataWithFeatures(bt.feeds.PandasData):
            lines = tuple(self.selected_features); params = (('volume', 'tick_volume'),) + tuple([(f, -1) for f in self.selected_features])
        cerebro = bt.Cerebro(stdstats=False); cerebro.adddata(PandasDataWithFeatures(dataname=df_fold))
        strategy_kwargs = {'model': model, 'features': self.selected_features, **self.strategy_params}
        cerebro.addstrategy(FinalMLStrategy, **strategy_kwargs); cerebro.broker.setcash(self.wfo_config['initial_cash']); cerebro.broker.setcommission(commission=self.wfo_config['commission'])
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades'); cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown'); cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        results = cerebro.run(); analysis = results[0].analyzers; trades_analysis = analysis.trades.get_analysis(); drawdown_analysis = analysis.drawdown.get_analysis(); sharpe_analysis = analysis.sharpe.get_analysis()
        if trades_analysis.total.total > 0:
            sharpe_ratio = sharpe_analysis.get('sharperatio')
            return {"pnl": trades_analysis.pnl.net.total, "total_trades": trades_analysis.total.total, "won_trades": trades_analysis.won.total, "lost_trades": trades_analysis.lost.total, "max_drawdown": drawdown_analysis.max.drawdown, "sharpe_ratio": sharpe_ratio if sharpe_ratio is not None else 0.0,}
        else:
            return {"pnl": 0.0, "total_trades": 0, "won_trades": 0, "lost_trades": 0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}

    # ★★★ 主市場處理函式：重大修改，改用新標籤 ★★★
    def run_for_single_market(self, market_file_path: Path):
        market_name = market_file_path.stem
        print(f"\n{'='*25} 開始處理市場: {market_name} {'='*25}")
        
        df = pd.read_parquet(market_file_path)
        df.index = pd.to_datetime(df.index)

        # ★★★ 使用三道門檻法創建 Label ★★★
        df = create_triple_barrier_labels(df, self.tb_settings)
        df.dropna(inplace=True)
        
        if df.empty:
            print(f"警告: 市場 {market_name} 在數據清洗後為空，已跳過。")
            return
            
        start_date = df.index.min(); end_date = df.index.max()
        train_days = timedelta(days=self.wfo_config['training_days']); val_days = timedelta(days=self.wfo_config['validation_days'])
        test_days = timedelta(days=self.wfo_config['testing_days']); step_days = timedelta(days=self.wfo_config['step_days'])
        current_date = start_date; fold_results = []; fold_number = 0

        while current_date + train_days + val_days + test_days <= end_date:
            fold_number += 1
            train_start = current_date; val_start = train_start + train_days
            test_start = val_start + val_days; test_end = test_start + test_days
            
            print(f"\n--- Fold {fold_number}: Train[{train_start.date()}-{val_start.date()}] | Val[{val_start.date()}-{test_start.date()}] | Test[{test_start.date()}-{test_end.date()}] ---")

            df_train = df[train_start:val_start]; df_val = df[val_start:test_start]; df_test = df[test_start:test_end]
            if df_train.empty or df_val.empty or df_test.empty:
                print("警告: 當前窗口數據不足，跳過此 Fold。"); current_date += step_days; continue

            X_train, y_train = df_train[self.selected_features], df_train['target']
            
            # ★★★ 執行 Optuna 超參數優化，傳入 df_val 供回測 ★★★
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, df_val), 
                           n_trials=self.wfo_config['n_trials'], show_progress_bar=True)
            print(f"資訊: 參數優化完成！最佳驗證集夏普比率: {study.best_value:.4f}")

            X_in_sample = pd.concat([df_train[self.selected_features], df_val[self.selected_features]])
            y_in_sample = pd.concat([df_train['target'], df_val['target']])
            final_model = lgb.LGBMClassifier(**study.best_params)
            final_model.fit(X_in_sample.values, y_in_sample.values)
            
            result = self.run_backtest_on_fold(df_test, final_model)
            fold_results.append(result)
            print(f"Fold {fold_number} 測試結果: PnL={result['pnl']:.2f}, Trades={result['total_trades']}, WinRate={(result['won_trades']/result['total_trades']*100 if result['total_trades']>0 else 0):.2f}%")
            
            current_date += step_days

        if not fold_results:
            print(f"\n市場 {market_name} 沒有足夠的數據來完成任何一次滾動回測。")
            return
            
        final_pnl = sum(r['pnl'] for r in fold_results); total_trades = sum(r['total_trades'] for r in fold_results)
        won_trades = sum(r['won_trades'] for r in fold_results); win_rate = (won_trades / total_trades) if total_trades > 0 else 0.0
        avg_max_drawdown = np.mean([r['max_drawdown'] for r in fold_results])
        valid_sharpes = [r['sharpe_ratio'] for r in fold_results if r['sharpe_ratio'] is not None]; avg_sharpe_ratio = np.mean(valid_sharpes) if valid_sharpes else 0.0
        
        print("\n" + f"--- {market_name} 滾動優化總結報告 ---"); print(f"  - 總淨利: {final_pnl:,.2f}"); print(f"  - 總交易次數: {total_trades}"); print(f"  - 總勝率: {win_rate:.2%}"); print(f"  - 平均最大回撤: {avg_max_drawdown:.2f}%"); print(f"  - 平均夏普比率: {avg_sharpe_ratio:.2f}"); print("-" * 50)
        self.all_market_results[market_name] = {"final_pnl": final_pnl, "total_trades": total_trades, "win_rate": win_rate}

    def run(self):
        # ... (此函式內容不變，為節省空間省略)
        print(f"========= 整合式滾動優化與回測流程開始 (版本 6.0) ========="); input_dir = Path(self.paths['features_data']); input_files = list(input_dir.rglob("*.parquet"))
        if not input_files: print(f"致命錯誤: 在 {input_dir} 中找不到任何數據檔案！"); return
        for market_file in sorted(input_files):
            try: self.run_for_single_market(market_file)
            except Exception as e: print(f"錯誤: 處理市場 {market_file.name} 時發生未預期的錯誤:"); traceback.print_exc()
        print("\n" + "="*25 + " 所有市場滾動回測最終總結 " + "="*25)
        if self.all_market_results:
            summary_df = pd.DataFrame.from_dict(self.all_market_results, orient='index'); summary_df.index.name = 'Market'; print("\n" + summary_df.to_string())
        else: print("資訊: 沒有任何市場完成回測。")
        print("=" * 70); print("========= 所有任務執行完畢，流程結束 =========")


if __name__ == "__main__":
    try:
        config = load_config()
        optimizer = MLOptimizerAndBacktester(config)
        optimizer.run()
    except Exception as e:
        print(f"腳本執行時發生未預期的嚴重錯誤:")
        traceback.print_exc()
        sys.exit(1)
