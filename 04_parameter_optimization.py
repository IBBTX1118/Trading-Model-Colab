# 檔名: 04_parameter_optimization.py
# 描述: Phase 2 實作：整合三道門檻標籤法與夏普比率優化。
# 版本: 6.3 (最終修正版 - 修復 Optuna objective KeyError 並篩選 D1)

import sys
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import timedelta
import traceback
import logging

import backtrader as bt
import lightgbm as lgb
import optuna

# (load_config, create_triple_barrier_labels, FinalMLStrategy 函式與之前相同，此處省略以保持簡潔)
def load_config(config_path: str = 'config.yaml') -> Dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)
    except FileNotFoundError: print(f"致命錯誤: 設定檔 {config_path} 不存在！"); sys.exit(1)
    except Exception as e: print(f"致命錯誤: 讀取設定檔 {config_path} 時發生錯誤: {e}"); sys.exit(1)

def create_triple_barrier_labels(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    df_out = df.copy(); tp_pct = settings['tp_pct']; sl_pct = settings['sl_pct']; max_hold = settings['max_hold_periods']
    outcomes = pd.DataFrame(index=df_out.index, columns=['hit_time', 'label'])
    for i in range(len(df_out) - max_hold):
        entry_price = df_out['close'].iloc[i]; tp_price = entry_price * (1 + tp_pct); sl_price = entry_price * (1 - sl_pct)
        window = df_out.iloc[i+1 : i+1+max_hold]
        hit_tp_time = window[window['high'] >= tp_price].index.min(); hit_sl_time = window[window['low'] <= sl_price].index.min()
        if pd.notna(hit_tp_time) and pd.notna(hit_sl_time):
            if hit_tp_time < hit_sl_time: outcomes.loc[df_out.index[i], 'label'] = 1; outcomes.loc[df_out.index[i], 'hit_time'] = hit_tp_time
            else: outcomes.loc[df_out.index[i], 'label'] = -1; outcomes.loc[df_out.index[i], 'hit_time'] = hit_sl_time
        elif pd.notna(hit_tp_time): outcomes.loc[df_out.index[i], 'label'] = 1; outcomes.loc[df_out.index[i], 'hit_time'] = hit_tp_time
        elif pd.notna(hit_sl_time): outcomes.loc[df_out.index[i], 'label'] = -1; outcomes.loc[df_out.index[i], 'hit_time'] = hit_sl_time
        else: outcomes.loc[df_out.index[i], 'label'] = 0; outcomes.loc[df_out.index[i], 'hit_time'] = window.index[-1]
    df_out = df_out.join(outcomes); df_out['target'] = (df_out['label'] == 1).astype(int); return df_out

class FinalMLStrategy(bt.Strategy):
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
                sl_price = current_price - self.p.atr_sl_multiplier * current_atr; tp_price = current_price + self.p.atr_tp_multiplier * current_atr
                self.buy_bracket(price=current_price, stopprice=sl_price, limitprice=tp_price,)

class MLOptimizerAndBacktester:
    def __init__(self, config: Dict):
        # ... (初始化內容不變，省略) ...
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout); formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter); self.logger.addHandler(handler); self.logger.setLevel(logging.INFO)
        self.config = config; self.paths = config['paths']; self.wfo_config = config['walk_forward_optimization']
        self.strategy_params = config['strategy_params']; self.tb_settings = config['triple_barrier_settings']
        self.output_base_dir = Path(self.paths['ml_pipeline_output']); self.output_base_dir.mkdir(parents=True, exist_ok=True)
        features_file = self.output_base_dir / self.paths['selected_features_filename']
        self.selected_features = self._load_json(features_file)['selected_features']
        self.all_market_results = {}; optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _load_json(self, file_path: Path) -> Dict:
        # ... (內容不變，省略) ...
        if not file_path.exists(): self.logger.error(f"致命錯誤: 輸入檔案 {file_path} 不存在！"); sys.exit(1)
        with open(file_path, 'r') as f: data = json.load(f)
        self.logger.info(f"成功從 {file_path} 載入 {len(data['selected_features'])} 個全域特徵。")
        return data

    # ★★★ 核心修正 #1：修正 objective 函式 ★★★
    def objective(self, trial: optuna.trial.Trial, X_train, y_train, df_val, available_features: list) -> float:
        """Optuna 的目標函數，優化驗證集上的夏普比率 (修正KeyError)"""
        param = { 'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'boosting_type': 'gbdt',
                  'n_estimators': trial.suggest_int('n_estimators', 100, 800), 'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                  'num_leaves': trial.suggest_int('num_leaves', 20, 200), 'max_depth': trial.suggest_int('max_depth', 3, 10),
                  'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True), 'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                  'seed': 42, 'n_jobs': -1, }
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train.values, y_train.values)
        
        # 在驗證集上執行迷你回測
        # run_backtest_on_fold 內部已有處理零交易的邏輯，直接調用即可
        result = self.run_backtest_on_fold(df_val, model, available_features)
        
        # 從回測結果中提取夏普比率
        sharpe = result.get('sharpe_ratio') 
        
        # 如果夏普為 0 或 None，給予一個負值懲罰，引導 Optuna 尋找能產生交易的參數
        if not sharpe or sharpe == 0.0:
            return -1.0
            
        return sharpe

    def run_backtest_on_fold(self, df_fold: pd.DataFrame, model: lgb.LGBMClassifier, available_features: list) -> Dict:
        # ... (此函式內容不變，省略) ...
        class PandasDataWithFeatures(bt.feeds.PandasData):
            lines = tuple(available_features); params = (('volume', 'tick_volume'),) + tuple([(f, -1) for f in available_features])
        cerebro = bt.Cerebro(stdstats=False); cerebro.adddata(PandasDataWithFeatures(dataname=df_fold))
        strategy_kwargs = {'model': model, 'features': available_features, **self.strategy_params}
        cerebro.addstrategy(FinalMLStrategy, **strategy_kwargs); cerebro.broker.setcash(self.wfo_config['initial_cash']); cerebro.broker.setcommission(commission=self.wfo_config['commission'])
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades'); cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown'); cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        results = cerebro.run(); analysis = results[0].analyzers; trades_analysis = analysis.trades.get_analysis(); drawdown_analysis = analysis.drawdown.get_analysis(); sharpe_analysis = analysis.sharpe.get_analysis()
        if trades_analysis.total.total > 0:
            sharpe_ratio = sharpe_analysis.get('sharperatio'); return {"pnl": trades_analysis.pnl.net.total, "total_trades": trades_analysis.total.total, "won_trades": trades_analysis.won.total, "lost_trades": trades_analysis.lost.total, "max_drawdown": drawdown_analysis.max.drawdown, "sharpe_ratio": sharpe_ratio if sharpe_ratio is not None else 0.0,}
        else: return {"pnl": 0.0, "total_trades": 0, "won_trades": 0, "lost_trades": 0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}

    def run_for_single_market(self, market_file_path: Path):
        # ... (此函式內容不變，省略) ...
        self.logger.info(f"{'='*25} 開始處理市場: {market_file_path.stem} {'='*25}"); df = pd.read_parquet(market_file_path); df.index = pd.to_datetime(df.index)
        available_features = [f for f in self.selected_features if f in df.columns]; self.logger.info(f"在 {market_file_path.stem} 中找到 {len(available_features)}/{len(self.selected_features)} 個可用特徵。")
        if len(available_features) < 5: self.logger.warning(f"可用特徵過少 (<5)，跳過市場 {market_file_path.stem}。"); return
        df = create_triple_barrier_labels(df, self.tb_settings); df.dropna(inplace=True)
        if df.empty: self.logger.warning(f"市場 {market_file_path.stem} 在數據清洗後為空，已跳過。"); return
        start_date, end_date = df.index.min(), df.index.max(); train_days, val_days = timedelta(days=self.wfo_config['training_days']), timedelta(days=self.wfo_config['validation_days']); test_days, step_days = timedelta(days=self.wfo_config['testing_days']), timedelta(days=self.wfo_config['step_days']); current_date, fold_results, fold_number = start_date, [], 0
        while current_date + train_days + val_days + test_days <= end_date:
            fold_number += 1; train_start, val_start = current_date, current_date + train_days; test_start, test_end = val_start + val_days, val_start + val_days + test_days
            print(f"\n--- Fold {fold_number}: Train[{train_start.date()}-{val_start.date()}] | Val[{val_start.date()}-{test_start.date()}] | Test[{test_start.date()}-{test_end.date()}] ---")
            df_train, df_val, df_test = df[train_start:val_start], df[val_start:test_start], df[test_start:test_end]
            if any(d.empty for d in [df_train, df_val, df_test]): self.logger.warning("當前窗口數據不足，跳過此 Fold。"); current_date += step_days; continue
            X_train, y_train = df_train[available_features], df_train['target']
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, df_val, available_features), n_trials=self.wfo_config['n_trials'], show_progress_bar=True)
            self.logger.info(f"參數優化完成！最佳驗證集夏普比率: {study.best_value:.4f}")
            X_in_sample = pd.concat([df_train[available_features], df_val[available_features]]); y_in_sample = pd.concat([df_train['target'], df_val['target']])
            final_model = lgb.LGBMClassifier(**study.best_params); final_model.fit(X_in_sample.values, y_in_sample.values)
            result = self.run_backtest_on_fold(df_test, final_model, available_features); fold_results.append(result)
            print(f"Fold {fold_number} 測試結果: PnL={result['pnl']:.2f}, Trades={result['total_trades']}, WinRate={(result['won_trades']/result['total_trades']*100 if result['total_trades']>0 else 0):.2f}%")
            current_date += step_days
        if not fold_results: self.logger.warning(f"市場 {market_file_path.stem} 沒有足夠數據完成滾動回測。"); return
        final_pnl = sum(r['pnl'] for r in fold_results); total_trades = sum(r['total_trades'] for r in fold_results); won_trades = sum(r['won_trades'] for r in fold_results); win_rate = (won_trades / total_trades) if total_trades > 0 else 0.0
        avg_max_drawdown = np.mean([r['max_drawdown'] for r in fold_results]); valid_sharpes = [r['sharpe_ratio'] for r in fold_results if r['sharpe_ratio'] is not None]; avg_sharpe_ratio = np.mean(valid_sharpes) if valid_sharpes else 0.0
        print("\n" + f"--- {market_file_path.stem} 滾動優化總結報告 ---"); print(f"  - 總淨利: {final_pnl:,.2f}"); print(f"  - 總交易次數: {total_trades}"); print(f"  - 總勝率: {win_rate:.2%}"); print(f"  - 平均最大回撤: {avg_max_drawdown:.2f}%"); print(f"  - 平均夏普比率: {avg_sharpe_ratio:.2f}"); print("-" * 50)
        self.all_market_results[market_file_path.stem] = {"final_pnl": final_pnl, "total_trades": total_trades, "win_rate": win_rate}

    # ★★★ 核心修正 #2：修改 run 函式以篩選 D1 數據 ★★★
    def run(self):
        self.logger.info(f"{'='*25} 整合式滾動優化與回測流程開始 (版本 6.3) {'='*25}")
        input_dir = Path(self.paths['features_data'])
        
        # 尋找所有 .parquet 檔案，然後進行篩選
        all_files = list(input_dir.rglob("*.parquet"))
        
        # 只保留檔名包含 "_D1.parquet" 的檔案
        input_files = [f for f in all_files if '_D1.parquet' in f.name]
        self.logger.info(f"已篩選出 {len(input_files)} 個 D1 市場檔案進行優先回測。")
        
        if not input_files: 
            self.logger.error(f"在 {input_dir} 中找不到任何 D1 數據檔案！"); return
        
        for market_file in sorted(input_files):
            try:
                self.run_for_single_market(market_file)
            except Exception:
                self.logger.error(f"處理市場 {market_file.name} 時發生未預期的錯誤:")
                traceback.print_exc()

        print("\n" + "="*25 + " 所有市場滾動回測最終總結 " + "="*25)
        if self.all_market_results:
            summary_df = pd.DataFrame.from_dict(self.all_market_results, orient='index'); summary_df.index.name = 'Market'
            print("\n" + summary_df.to_string())
        else:
            self.logger.info("沒有任何市場完成回測。")
        self.logger.info(f"{'='*30} 所有任務執行完畢 {'='*30}")


if __name__ == "__main__":
    try:
        config = load_config()
        optimizer = MLOptimizerAndBacktester(config)
        optimizer.run()
    except Exception:
        print(f"腳本執行時發生未預期的嚴重錯誤:")
        traceback.print_exc()
        sys.exit(1)
