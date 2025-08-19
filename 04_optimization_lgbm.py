# 檔名: 04_optimization_lgbm.py
# 描述: 使用 LightGBM 模型進行參數優化與回測。
# 版本: 13.0 (LGBM 基準版)

import sys; import yaml; import json; from pathlib import Path; from typing import Dict
import pandas as pd; import numpy as np; from datetime import timedelta; import traceback; import logging
import backtrader as bt; import lightgbm as lgb; import optuna

# ==============================================================================
#                      輔助函式 (無變動)
# ==============================================================================
def load_config(config_path: str = 'config.yaml') -> Dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)
    except FileNotFoundError: print(f"致命錯誤: 設定檔 {config_path} 不存在！"); sys.exit(1)
    except Exception as e: print(f"致命錯誤: 讀取設定檔 {config_path} 時發生錯誤: {e}"); sys.exit(1)

def create_triple_barrier_labels(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    df_out = df.copy(); tp_multiplier = settings['tp_atr_multiplier']; sl_multiplier = settings['sl_atr_multiplier']; max_hold = settings['max_hold_periods']
    atr_col_name = 'D1_ATR_14' if 'D1_ATR_14' in df_out.columns else 'ATR_14'
    if atr_col_name not in df_out.columns: raise ValueError(f"數據中缺少 ATR 欄位，無法創建標籤。")
    outcomes = pd.DataFrame(index=df_out.index, columns=['label'])
    high_series, low_series, atr_series = df_out['high'], df_out['low'], df_out[atr_col_name]
    for i in range(len(df_out) - max_hold):
        entry_price, atr_at_entry = df_out['close'].iloc[i], atr_series.iloc[i]
        if atr_at_entry <= 0 or pd.isna(atr_at_entry): continue
        tp_price = entry_price + (atr_at_entry * tp_multiplier); sl_price = entry_price - (atr_at_entry * sl_multiplier)
        window = df_out.iloc[i+1 : i+1+max_hold]
        hit_tp_time = window[high_series.iloc[i+1:i+1+max_hold] >= tp_price].index.min(); hit_sl_time = window[low_series.iloc[i+1:i+1+max_hold] <= sl_price].index.min()
        if pd.notna(hit_tp_time) and pd.notna(hit_sl_time): outcomes.loc[df_out.index[i], 'label'] = 1 if hit_tp_time < hit_sl_time else -1
        elif pd.notna(hit_tp_time): outcomes.loc[df_out.index[i], 'label'] = 1
        elif pd.notna(hit_sl_time): outcomes.loc[df_out.index[i], 'label'] = -1
        else: outcomes.loc[df_out.index[i], 'label'] = 0
    df_out = df_out.join(outcomes[['label']]); df_out['target'] = (df_out['label'] == 1).astype(int); return df_out

# ==============================================================================
#                      交易策略 (無變動)
# ==============================================================================
class FinalMLStrategy(bt.Strategy):
    params = (('model', None), ('features', None), ('entry_threshold', 0.45), ('tp_atr_multiplier', 2.5), ('sl_atr_multiplier', 1.5),)
    def __init__(self):
        if not self.p.model or not self.p.features: raise ValueError("模型和特徵列表必須提供！")
        self.feature_lines = [getattr(self.data.lines, f) for f in self.p.features]
        d1_uptrend_name = 'D1_is_uptrend' if hasattr(self.data.lines, 'D1_is_uptrend') else 'is_uptrend'
        self.is_uptrend = getattr(self.data.lines, d1_uptrend_name, lambda: True)
        atr_name = 'D1_ATR_14' if hasattr(self.data.lines, 'D1_ATR_14') else 'ATR_14'
        self.atr = getattr(self.data.lines, atr_name)
    def next(self):
        if self.position: return
        try: feature_vector = np.array([line[0] for line in self.feature_lines]).reshape(1, -1)
        except IndexError: return
        pred_probs = self.p.model.predict_proba(feature_vector)[0]
        prob_sl, prob_tp, prob_hold = pred_probs[0], pred_probs[1], pred_probs[2]
        current_price, atr_value = self.data.close[0], self.atr[0]
        if atr_value <= 0: return
        market_is_uptrend = self.is_uptrend[0] > 0.5 
        if market_is_uptrend and prob_tp > prob_sl and prob_tp > self.p.entry_threshold:
            sl_dist = atr_value * self.p.sl_atr_multiplier; tp_dist = atr_value * self.p.tp_atr_multiplier
            self.buy_bracket(price=current_price, stopprice=current_price - sl_dist, limitprice=current_price + tp_dist)
        elif not market_is_uptrend and prob_sl > prob_tp and prob_sl > self.p.entry_threshold:
            sl_dist = atr_value * self.p.sl_atr_multiplier; tp_dist = atr_value * self.p.tp_atr_multiplier
            self.sell_bracket(price=current_price, stopprice=current_price + sl_dist, limitprice=current_price - tp_dist)

# ==============================================================================
#                      優化器與回測器
# ==============================================================================
class MLOptimizerAndBacktester:
    def __init__(self, config: Dict):
        self.config = config; self.paths = config['paths']; self.wfo_config = config['walk_forward_optimization']
        self.strategy_params = config.get('strategy_params', {}); self.tb_settings = config['triple_barrier_settings']
        self.strategy_params['tp_atr_multiplier'] = self.tb_settings.get('tp_atr_multiplier', 2.5)
        self.strategy_params['sl_atr_multiplier'] = self.tb_settings.get('sl_atr_multiplier', 1.5)
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout); formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter); self.logger.addHandler(handler); self.logger.setLevel(logging.INFO)
        self.output_base_dir = Path(self.paths['ml_pipeline_output']); self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.all_market_results = {}; optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _load_json(self, file_path: Path) -> Dict:
        if not file_path.exists(): self.logger.error(f"致命錯誤: 輸入檔案 {file_path} 不存在！"); return {}
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        self.logger.info(f"成功從 {file_path} 載入 {len(data.get('selected_features', []))} 個專屬特徵。")
        return data

    def _save_json(self, data: Dict, file_path: Path):
        try:
            with open(file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4, ensure_ascii=False)
            self.logger.info(f"成功將數據儲存至: {file_path}")
        except Exception as e: self.logger.error(f"儲存 JSON 檔案至 {file_path} 時發生錯誤: {e}")

    def objective(self, trial: optuna.trial.Trial, X_train, y_train, df_val, available_features: list, market_name: str) -> float:
        # LightGBM 參數
        model_param = {
            'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 3, 'verbosity': -1,
            'boosting_type': 'gbdt', 'seed': 42, 'n_jobs': -1,
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        }
        
        # 策略參數
        strategy_param_updates = {
            'entry_threshold': trial.suggest_float('entry_threshold', 0.35, 0.55, step=0.01),
            'tp_atr_multiplier': trial.suggest_float('tp_atr_multiplier', 1.5, 4.0),
            'sl_atr_multiplier': trial.suggest_float('sl_atr_multiplier', 1.0, 2.5),
        }
        if strategy_param_updates['tp_atr_multiplier'] <= strategy_param_updates['sl_atr_multiplier']: return -999.0
        
        # 訓練與回測
        model = lgb.LGBMClassifier(**model_param)
        model.fit(X_train, y_train)
        temp_strategy_params = {**self.strategy_params, **strategy_param_updates}
        result = self.run_backtest_on_fold(df_val, model, available_features, temp_strategy_params)
        
        min_trades_threshold = 10
        if result.get('total_trades', 0) < min_trades_threshold: return -999.0

        sharpe = result.get('sharpe_ratio', -1.0)
        return sharpe if sharpe is not None else -1.0
    
    def run_backtest_on_fold(self, df_fold, model, available_features, strategy_params_override=None):
        all_feature_columns = [c for c in df_fold.columns if c not in ['open','high','low','close','tick_volume','spread','real_volume','label','target_multiclass','hit_time']]
        class PandasDataWithFeatures(bt.feeds.PandasData):
            lines=tuple(all_feature_columns); params=(('volume','tick_volume'),)+tuple([(c,-1) for c in all_feature_columns])
        cerebro=bt.Cerebro(stdstats=False); cerebro.adddata(PandasDataWithFeatures(dataname=df_fold))
        final_strategy_params=strategy_params_override if strategy_params_override is not None else self.strategy_params
        strategy_kwargs={'model':model,'features':available_features,**final_strategy_params}; cerebro.addstrategy(FinalMLStrategy,**strategy_kwargs)
        cerebro.broker.setcash(self.wfo_config['initial_cash']); cerebro.broker.setcommission(commission=self.wfo_config['commission'])
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer,_name='trades'); cerebro.addanalyzer(bt.analyzers.DrawDown,_name='drawdown')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio,_name='sharpe'); cerebro.addanalyzer(bt.analyzers.SQN,_name='sqn')
        try:
            results=cerebro.run(); analysis=results[0].analyzers; trades_analysis=analysis.trades.get_analysis()
            if trades_analysis.get('total',{}).get('total',0)>0:
                sharpe_ratio=analysis.sharpe.get_analysis().get('sharperatio')
                return {"pnl":trades_analysis.pnl.net.total or 0.0, "total_trades":trades_analysis.total.total,"won_trades":trades_analysis.won.total or 0,"lost_trades":trades_analysis.lost.total or 0,"pnl_won_total":trades_analysis.won.pnl.total or 0.0,"pnl_lost_total":trades_analysis.lost.pnl.total or 0.0,"sqn":analysis.sqn.get_analysis().get('sqn'),"max_drawdown":analysis.drawdown.get_analysis().max.drawdown,"sharpe_ratio":sharpe_ratio if sharpe_ratio is not None else 0.0,}
        except Exception as e: self.logger.error(f"回測期間發生錯誤: {e}",exc_info=False)
        return {"pnl":0.0,"total_trades":0,"won_trades":0,"lost_trades":0,"max_drawdown":0.0,"sharpe_ratio":0.0,"pnl_won_total":0.0,"pnl_lost_total":0.0,"sqn":0.0,"profit_factor":0.0}

    def run_for_single_market(self, market_file_path: Path):
        market_name = market_file_path.stem
        self.logger.info(f"{'='*25} 開始處理市場: {market_name} {'='*25}")
        
        features_filename = self.output_base_dir / f"selected_features_{market_name}.json"
        features_data = self._load_json(features_filename)
        if not features_data: self.logger.warning(f"找不到市場 {market_name} 的特徵檔案，跳過。"); return
        selected_features = features_data['selected_features']

        df = pd.read_parquet(market_file_path); df.index = pd.to_datetime(df.index)
        uptrend_feature_name = 'D1_is_uptrend' if '_H' in market_name else 'is_uptrend'
        if uptrend_feature_name not in df.columns: self.logger.warning(f"重要特徵 '{uptrend_feature_name}' 不存在，跳過。"); return
        
        available_features = [f for f in selected_features if f in df.columns]
        self.logger.info(f"在 {market_name} 中找到 {len(available_features)}/{len(selected_features)} 個可用專屬特徵。")
        if len(available_features) < 5: self.logger.warning(f"可用特徵過少 (<5)，跳過。"); return

        df = create_triple_barrier_labels(df, self.tb_settings)
        mapping = {1: 1, -1: 0, 0: 2}; df['target_multiclass'] = df['label'].map(mapping); df.dropna(inplace=True)
        if df.empty: self.logger.warning(f"數據清洗後為空，跳過。"); return
            
        start_date, end_date = df.index.min(), df.index.max()
        wfo_days = {k: timedelta(days=self.wfo_config[k]) for k in ['training_days', 'validation_days', 'testing_days', 'step_days']}
        current_date, fold_results, fold_number = start_date, [], 0
        all_fold_best_params = []

        while current_date + wfo_days['training_days'] + wfo_days['validation_days'] + wfo_days['testing_days'] <= end_date:
            fold_number += 1
            train_start, val_start, test_start, test_end = current_date, current_date + wfo_days['training_days'], current_date + wfo_days['training_days'] + wfo_days['validation_days'], current_date + wfo_days['training_days'] + wfo_days['validation_days'] + wfo_days['testing_days']
            print(f"\n--- Fold {fold_number}: Train[{train_start.date()}-{val_start.date()}] | Val[{val_start.date()}-{test_start.date()}] | Test[{test_start.date()}-{test_end.date()}] ---")
            
            df_train, df_val, df_test = df.loc[train_start:val_start], df.loc[val_start:test_start], df.loc[test_start:test_end]
            if any(d.empty for d in [df_train, df_val, df_test]): self.logger.warning("當前窗口數據不足，跳過。"); current_date += wfo_days['step_days']; continue
            
            X_train, y_train = df_train[available_features], df_train['target_multiclass']
            
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, df_val, available_features, market_name), n_trials=self.wfo_config.get('n_trials', 50), show_progress_bar=True)
            
            self.logger.info(f"優化完成！最佳驗證集夏普: {study.best_value:.4f}"); self.logger.info(f"最佳參數: {study.best_params}")
            params_with_fold = {'fold': fold_number, 'best_sharpe_in_val': study.best_value, **study.best_params}; all_fold_best_params.append(params_with_fold)

            X_in_sample = pd.concat([df_train[available_features], df_val[available_features]])
            y_in_sample = pd.concat([df_train['target_multiclass'], df_val['target_multiclass']])
            
            model_params = {k: v for k, v in study.best_params.items() if k not in self.strategy_params.keys() and k not in ['entry_threshold', 'tp_atr_multiplier', 'sl_atr_multiplier']}
            model_params.update({'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 3, 'verbosity': -1, 'seed': 42})
            
            final_model = lgb.LGBMClassifier(**model_params)
            final_model.fit(X_in_sample, y_in_sample)
            
            final_test_params = self.strategy_params.copy()
            for k in ['entry_threshold', 'tp_atr_multiplier', 'sl_atr_multiplier']:
                final_test_params[k] = study.best_params.get(k, self.strategy_params[k])
            
            result = self.run_backtest_on_fold(df_test, final_model, available_features, final_test_params); fold_results.append(result)
            win_rate = (result['won_trades'] / result['total_trades'] * 100 if result['total_trades'] > 0 else 0)
            print(f"Fold {fold_number} 測試結果: PnL={result['pnl']:.2f}, Trades={result['total_trades']}, WinRate={win_rate:.2f}% (使用最佳參數)")
            current_date += wfo_days['step_days']
        
        params_filename = self.output_base_dir / f"{market_name}_best_params_lgbm.json"
        self._save_json({"market": market_name, "folds_data": all_fold_best_params}, params_filename)
        
        if not fold_results: self.logger.warning(f"{market_name} 沒有足夠數據完成回測。"); return
            
        final_pnl = sum(r['pnl'] for r in fold_results); total_trades = sum(r['total_trades'] for r in fold_results); won_trades = sum(r['won_trades'] for r in fold_results)
        total_won_pnl = sum(r['pnl_won_total'] for r in fold_results); total_lost_pnl = sum(r['pnl_lost_total'] for r in fold_results)
        profit_factor = total_won_pnl / abs(total_lost_pnl) if total_lost_pnl != 0 else float('inf')
        win_rate = (won_trades / total_trades) if total_trades > 0 else 0.0
        avg_max_drawdown = np.mean([r['max_drawdown'] for r in fold_results])
        valid_sharpes = [r['sharpe_ratio'] for r in fold_results if r['sharpe_ratio'] is not None and r['sharpe_ratio'] != 0.0]
        avg_sharpe_ratio = np.mean(valid_sharpes) if valid_sharpes else 0.0
        valid_sqns = [r['sqn'] for r in fold_results if r.get('sqn') is not None]; avg_sqn = np.mean(valid_sqns) if valid_sqns else 0.0
        
        report = (f"\n--- {market_name} (LightGBM) 滾動優化總結報告 ---\n"
                  f"  - 總淨利: {final_pnl:,.2f}\n  - 總交易次數: {total_trades}\n"
                  f"  - 總勝率: {win_rate:.2%}\n  - 獲利因子: {profit_factor:.2f}\n"
                  f"  - 平均 SQN: {avg_sqn:.2f}\n  - 平均最大回撤: {avg_max_drawdown:.2f}%\n"
                  f"  - 平均夏普比率: {avg_sharpe_ratio:.2f}\n"
                  f"  - 詳細參數已儲存至: {params_filename.name}\n{'-'*50}")
        print(report)
        self.all_market_results[market_file_path.stem] = {"final_pnl": final_pnl, "total_trades": total_trades, "win_rate": win_rate, "profit_factor": profit_factor, "avg_sqn": avg_sqn, "avg_sharpe": avg_sharpe_ratio}

    def run(self):
        self.logger.info(f"{'='*25} LightGBM 整合式滾動優化與回測流程開始 (版本 13.0) {'='*25}")
        input_dir = Path(self.paths['features_data'])
        all_files = list(input_dir.rglob("*.parquet")); input_files = [f for f in all_files if '_H4.parquet' in f.name]
        self.logger.info(f"已篩選出 {len(input_files)} 個 H4 市場檔案進行回測。")
        if not input_files: self.logger.error(f"在 {input_dir} 中找不到任何 H4 數據檔案！"); return
        for market_file in sorted(input_files):
            try: self.run_for_single_market(market_file)
            except Exception: self.logger.error(f"處理市場 {market_file.name} 時發生未預期的錯誤:"); traceback.print_exc()
        print("\n" + "="*25 + " 所有市場滾動回測最終總結 (LightGBM) " + "="*25)
        if self.all_market_results:
            summary_df = pd.DataFrame.from_dict(self.all_market_results, orient='index'); summary_df.index.name = 'Market'
            cols_order = ['final_pnl', 'total_trades', 'win_rate', 'profit_factor', 'avg_sqn', 'avg_sharpe']
            summary_df = summary_df[[col for col in cols_order if col in summary_df.columns]]
            print("\n" + summary_df.to_string(float_format="%.4f"))
        else: self.logger.info("沒有任何市場完成回測。")
        self.logger.info(f"{'='*30} 所有任務執行完畢 {'='*30}")

if __name__ == "__main__":
    try:
        config = load_config()
        optimizer = MLOptimizerAndBacktester(config)
        optimizer.run()
    except Exception as e:
        print(f"腳本執行時發生未預期的嚴重錯誤:")
        traceback.print_exc()
        sys.exit(1)
