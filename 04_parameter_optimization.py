# 檔名: 04_parameter_optimization.py
# 描述: Phase 2.3 實作：將策略參數加入 Optuna 優化。
# 版本: 8.0 (優化策略參數)

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

# ==============================================================================
#                      輔助函式
# ==============================================================================

def load_config(config_path: str = 'config.yaml') -> Dict:
    """載入 YAML 設定檔"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"致命錯誤: 設定檔 {config_path} 不存在！")
        sys.exit(1)
    except Exception as e:
        print(f"致命錯誤: 讀取設定檔 {config_path} 時發生錯誤: {e}")
        sys.exit(1)

def create_triple_barrier_labels(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """為 DataFrame 創建三道門檻標籤。"""
    df_out = df.copy()
    tp_pct = settings['tp_pct']
    sl_pct = settings['sl_pct']
    max_hold = settings['max_hold_periods']
    
    outcomes = pd.DataFrame(index=df_out.index, columns=['hit_time', 'label'])
    
    high_series = df_out['high']
    low_series = df_out['low']
    
    for i in range(len(df_out) - max_hold):
        entry_price = df_out['close'].iloc[i]
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
        
        window = df_out.iloc[i+1 : i+1+max_hold]
        
        hit_tp_time = window[high_series.iloc[i+1:i+1+max_hold] >= tp_price].index.min()
        hit_sl_time = window[low_series.iloc[i+1:i+1+max_hold] <= sl_price].index.min()
        
        if pd.notna(hit_tp_time) and pd.notna(hit_sl_time):
            if hit_tp_time < hit_sl_time:
                outcomes.loc[df_out.index[i], 'label'] = 1
            else:
                outcomes.loc[df_out.index[i], 'label'] = -1
        elif pd.notna(hit_tp_time):
            outcomes.loc[df_out.index[i], 'label'] = 1
        elif pd.notna(hit_sl_time):
            outcomes.loc[df_out.index[i], 'label'] = -1
        else:
            outcomes.loc[df_out.index[i], 'label'] = 0
            
    df_out = df_out.join(outcomes[['label']])
    return df_out

# ==============================================================================
#                      交易策略 (無變動)
# ==============================================================================
class FinalMLStrategy(bt.Strategy):
    params = (
        ('model', None),
        ('features', None),
        ('entry_threshold', 0.45),
        ('tp_pct', 0.015),
        ('sl_pct', 0.01),
    )
    
    def __init__(self):
        if not self.p.model or not self.p.features:
            raise ValueError("模型和特徵列表必須提供！")
        self.feature_lines = [getattr(self.data.lines, f) for f in self.p.features]
        self.is_uptrend = getattr(self.data.lines, 'is_uptrend', lambda: True)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        # print(f'{dt.isoformat()} - {txt}')

    def next(self):
        if self.position:
            return

        try:
            feature_vector = np.array([line[0] for line in self.feature_lines]).reshape(1, -1)
        except IndexError:
            return
            
        pred_probs = self.p.model.predict_proba(feature_vector)[0]
        prob_sl = pred_probs[0]
        prob_tp = pred_probs[1]
        current_price = self.data.close[0]
        
        market_is_uptrend = self.is_uptrend[0] > 0.5 

        if market_is_uptrend and prob_tp > prob_sl and prob_tp > self.p.entry_threshold:
            sl_price = current_price * (1 - self.p.sl_pct)
            tp_price = current_price * (1 + self.p.tp_pct)
            self.buy_bracket(price=current_price, stopprice=sl_price, limitprice=tp_price)
            self.log(f"趨勢向上，建立買單 @ {current_price:.5f}")

        elif not market_is_uptrend and prob_sl > prob_tp and prob_sl > self.p.entry_threshold:
            sl_price = current_price * (1 + self.p.sl_pct)
            tp_price = current_price * (1 - self.p.tp_pct)
            self.sell_bracket(price=current_price, stopprice=sl_price, limitprice=tp_price)
            self.log(f"趨勢向下，建立賣單 @ {current_price:.5f}")

# ==============================================================================
#                      優化器與回測器
# ==============================================================================
class MLOptimizerAndBacktester:
    def __init__(self, config: Dict):
        # ... __init__ 函式無變動 ...
        self.config = config
        self.paths = config['paths']
        self.wfo_config = config['walk_forward_optimization']
        self.strategy_params = config.get('strategy_params', {})
        self.tb_settings = config['triple_barrier_settings']
        self.strategy_params['tp_pct'] = self.tb_settings['tp_pct']
        self.strategy_params['sl_pct'] = self.tb_settings['sl_pct']
        
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.output_base_dir = Path(self.paths['ml_pipeline_output'])
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        features_file = self.output_base_dir / self.paths['selected_features_filename']
        self.selected_features = self._load_json(features_file)['selected_features']
        self.all_market_results = {}
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _load_json(self, file_path: Path) -> Dict:
        # ... _load_json 函式無變動 ...
        if not file_path.exists():
            self.logger.error(f"致命錯誤: 輸入檔案 {file_path} 不存在！")
            sys.exit(1)
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.logger.info(f"成功從 {file_path} 載入 {len(data.get('selected_features', []))} 個全域特徵。")
        return data

    def objective(self, trial: optuna.trial.Trial, X_train, y_train, df_val, available_features: list) -> float:
        # ★★★ 變動點 1: 優化目標新增 entry_threshold ★★★
        param = {
            'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 3,
            'verbosity': -1, 'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'seed': 42, 'n_jobs': -1,
        }
        
        # 讓 Optuna 建議一個最佳的進場門檻
        entry_threshold_opt = trial.suggest_float('entry_threshold', 0.40, 0.65, step=0.01)

        model = lgb.LGBMClassifier(**param)
        model.fit(X_train.values, y_train.values)
        
        # 建立一個臨時的策略參數字典，傳入 Optuna 建議的值
        temp_strategy_params = self.strategy_params.copy()
        temp_strategy_params['entry_threshold'] = entry_threshold_opt
        
        # 將這個臨時參數字典傳遞給回測函式
        result = self.run_backtest_on_fold(df_val, model, available_features, temp_strategy_params)
        
        sharpe = result.get('sharpe_ratio', -1.0)
        return sharpe if sharpe is not None else -1.0

    # ★★★ 變動點 2: 修改函式簽名以接收動態參數 ★★★
    def run_backtest_on_fold(self, df_fold: pd.DataFrame, model: lgb.LGBMClassifier, available_features: list, strategy_params_override: Dict = None) -> Dict:
        all_feature_columns = [
            col for col in df_fold.columns 
            if col not in ['open', 'high', 'low', 'close', 'tick_volume', 
                           'spread', 'real_volume', 'label', 'target_multiclass', 'hit_time']
        ]

        class PandasDataWithFeatures(bt.feeds.PandasData):
            lines = tuple(all_feature_columns)
            params = (('volume', 'tick_volume'),) + tuple([(col, -1) for col in all_feature_columns])

        cerebro = bt.Cerebro(stdstats=False)
        cerebro.adddata(PandasDataWithFeatures(dataname=df_fold))
        
        # 如果有傳入覆蓋參數(來自Optuna)，就使用它，否則使用 self.strategy_params (來自config)
        final_strategy_params = strategy_params_override if strategy_params_override is not None else self.strategy_params
        
        strategy_kwargs = {
            'model': model, 
            'features': available_features,
            **final_strategy_params
        }
        cerebro.addstrategy(FinalMLStrategy, **strategy_kwargs)
        
        cerebro.broker.setcash(self.wfo_config['initial_cash'])
        cerebro.broker.setcommission(commission=self.wfo_config['commission'])
        
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        
        try:
            results = cerebro.run()
            analysis = results[0].analyzers
            trades_analysis = analysis.trades.get_analysis()
            drawdown_analysis = analysis.drawdown.get_analysis()
            sharpe_analysis = analysis.sharpe.get_analysis()
            
            if trades_analysis.get('total', {}).get('total', 0) > 0:
                sharpe_ratio = sharpe_analysis.get('sharperatio')
                return {"pnl": trades_analysis.pnl.net.total, "total_trades": trades_analysis.total.total, "won_trades": trades_analysis.won.total, "lost_trades": trades_analysis.lost.total, "max_drawdown": drawdown_analysis.max.drawdown, "sharpe_ratio": sharpe_ratio if sharpe_ratio is not None else 0.0}
        except Exception as e:
            self.logger.error(f"回測期間發生錯誤: {e}", exc_info=False)

        return {"pnl": 0.0, "total_trades": 0, "won_trades": 0, "lost_trades": 0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}

    def run_for_single_market(self, market_file_path: Path):
        # ... run_for_single_market 前半部分無變動 ...
        self.logger.info(f"{'='*25} 開始處理市場: {market_file_path.stem} {'='*25}")
        df = pd.read_parquet(market_file_path)
        df.index = pd.to_datetime(df.index)
        
        if 'is_uptrend' not in df.columns:
            self.logger.warning(f"重要特徵 'is_uptrend' 不存在於 {market_file_path.stem} 的數據欄位中，跳過此市場。請確認是否已執行 02 號腳本。")
            return

        available_features = [f for f in self.selected_features if f in df.columns]
        self.logger.info(f"在 {market_file_path.stem} 中找到 {len(available_features)}/{len(self.selected_features)} 個可用特徵。")
        
        if len(available_features) < 5:
            self.logger.warning(f"可用特徵過少 (<5)，跳過市場 {market_file_path.stem}。")
            return

        df = create_triple_barrier_labels(df, self.tb_settings)
        mapping = {1: 1, -1: 0, 0: 2}
        df['target_multiclass'] = df['label'].map(mapping)
        df.dropna(inplace=True)

        if df.empty:
            self.logger.warning(f"市場 {market_file_path.stem} 在數據清洗後為空，已跳過。")
            return
            
        start_date, end_date = df.index.min(), df.index.max()
        train_days, val_days = timedelta(days=self.wfo_config['training_days']), timedelta(days=self.wfo_config['validation_days'])
        test_days, step_days = timedelta(days=self.wfo_config['testing_days']), timedelta(days=self.wfo_config['step_days'])
        current_date, fold_results, fold_number = start_date, [], 0
        
        while current_date + train_days + val_days + test_days <= end_date:
            fold_number += 1
            train_start, val_start = current_date, current_date + train_days
            test_start, test_end = val_start + val_days, val_start + val_days + test_days
            print(f"\n--- Fold {fold_number}: Train[{train_start.date()}-{val_start.date()}] | Val[{val_start.date()}-{test_start.date()}] | Test[{test_start.date()}-{test_end.date()}] ---")
            
            df_train, df_val, df_test = df.loc[train_start:val_start], df.loc[val_start:test_start], df.loc[test_start:test_end]
            
            if any(d.empty for d in [df_train, df_val, df_test]):
                self.logger.warning("當前窗口數據不足，跳過此 Fold。"); current_date += step_days; continue
                
            X_train, y_train = df_train[available_features], df_train['target_multiclass']
            
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, df_val, available_features), n_trials=self.wfo_config.get('n_trials', 50), show_progress_bar=True)
            
            self.logger.info(f"參數優化完成！最佳驗證集夏普比率: {study.best_value:.4f}")
            self.logger.info(f"找到的最佳參數: {study.best_params}")

            X_in_sample = pd.concat([df_train[available_features], df_val[available_features]])
            y_in_sample = pd.concat([df_train['target_multiclass'], df_val['target_multiclass']])
            
            # 從 study.best_params 中提取模型參數
            model_params = {k: v for k, v in study.best_params.items() if k != 'entry_threshold'}
            model_params.update({'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 3, 'verbosity': -1})
            final_model = lgb.LGBMClassifier(**model_params)
            final_model.fit(X_in_sample.values, y_in_sample.values)
            
            # ★★★ 變動點 3: 建立最終測試用的策略參數字典 ★★★
            # 獲取優化找到的最佳 entry_threshold，如果找不到則使用 config 中的預設值
            best_entry_threshold = study.best_params.get('entry_threshold', self.strategy_params.get('entry_threshold', 0.5))
            final_test_params = self.strategy_params.copy()
            final_test_params['entry_threshold'] = best_entry_threshold

            # 將最佳參數應用於最終測試
            result = self.run_backtest_on_fold(df_test, final_model, available_features, final_test_params)
            fold_results.append(result)
            win_rate = (result['won_trades'] / result['total_trades'] * 100 if result['total_trades'] > 0 else 0)
            print(f"Fold {fold_number} 測試結果: PnL={result['pnl']:.2f}, Trades={result['total_trades']}, WinRate={win_rate:.2f}% (使用最佳門檻: {best_entry_threshold:.2f})")
            
            current_date += step_days
        
        # ... 總結報告部分無變動 ...
        if not fold_results: 
            self.logger.warning(f"市場 {market_file_path.stem} 沒有足夠數據完成滾動回測。"); return
            
        final_pnl, total_trades, won_trades = sum(r['pnl'] for r in fold_results), sum(r['total_trades'] for r in fold_results), sum(r['won_trades'] for r in fold_results)
        win_rate = (won_trades / total_trades) if total_trades > 0 else 0.0
        avg_max_drawdown = np.mean([r['max_drawdown'] for r in fold_results])
        valid_sharpes = [r['sharpe_ratio'] for r in fold_results if r['sharpe_ratio'] is not None and r['sharpe_ratio'] != 0.0]
        avg_sharpe_ratio = np.mean(valid_sharpes) if valid_sharpes else 0.0
        
        report = (f"\n--- {market_file_path.stem} 滾動優化總結報告 ---\n"
                  f"  - 總淨利: {final_pnl:,.2f}\n" f"  - 總交易次數: {total_trades}\n"
                  f"  - 總勝率: {win_rate:.2%}\n" f"  - 平均最大回撤: {avg_max_drawdown:.2f}%\n"
                  f"  - 平均夏普比率: {avg_sharpe_ratio:.2f}\n" f"{'-'*50}")
        print(report)
        self.all_market_results[market_file_path.stem] = {"final_pnl": final_pnl, "total_trades": total_trades, "win_rate": win_rate, "avg_sharpe": avg_sharpe_ratio}

    def run(self):
        # ... run 函式無變動 ...
        self.logger.info(f"{'='*25} 整合式滾動優化與回測流程開始 (版本 8.0) {'='*25}")
        input_dir = Path(self.paths['features_data'])
        all_files = list(input_dir.rglob("*.parquet"))
        input_files = [f for f in all_files if '_D1.parquet' in f.name]
        self.logger.info(f"已篩選出 {len(input_files)} 個 D1 市場檔案進行優先回測。")
        
        if not input_files: self.logger.error(f"在 {input_dir} 中找不到任何 D1 數據檔案！"); return
        
        for market_file in sorted(input_files):
            try:
                self.run_for_single_market(market_file)
            except Exception:
                self.logger.error(f"處理市場 {market_file.name} 時發生未預期的錯誤:"); traceback.print_exc()

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
    except Exception as e:
        print(f"腳本執行時發生未預期的嚴重錯誤:"); traceback.print_exc(); sys.exit(1)
