# 檔名: 04_parameter_optimization.py
# 描述: Phase 2.1 實作：升級為多分類模型，統一出口，實現多空交易。
# 版本: 7.0 (多分類模型 + 統一出口)

import sys
import yaml
import json
from pathlib import Path
# ... 其他 import 維持不變 ...
import pandas as pd
import numpy as np
from datetime import timedelta
import traceback
import logging
import backtrader as bt
import lightgbm as lgb
import optuna

# (load_config 和 create_triple_barrier_labels 函式維持不變)
# ...

# ==============================================================================
# ★★★ 全面升級的交易策略 ★★★
# ==============================================================================
class FinalMLStrategy(bt.Strategy):
    params = (
        ('model', None),
        ('features', None),
        ('entry_threshold', 0.45), # 機率門檻，可設為優化參數
        ('tp_pct', 0.015),         # 停利百分比
        ('sl_pct', 0.01),          # 停損百分比
    )
    
    def __init__(self):
        if not self.p.model or not self.p.features:
            raise ValueError("模型和特徵列表必須提供！")
        # 將特徵從數據流中提取出來
        self.feature_lines = [getattr(self.data.lines, f) for f in self.p.features]
        # 記錄日誌
        self.log(f"策略初始化完成。Entry Threshold: {self.p.entry_threshold}, TP: {self.p.tp_pct:.2%}, SL: {self.p.sl_pct:.2%}")

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')

    def next(self):
        # 如果已經有倉位，則不執行任何操作
        if self.position:
            return

        try:
            # 準備當前 K 棒的特徵向量
            feature_vector = np.array([line[0] for line in self.feature_lines]).reshape(1, -1)
        except IndexError:
            # 數據還未就緒
            return
            
        # 使用多分類模型預測機率
        # 預期輸出 shape: (1, 3)，順序為 [P(SL), P(TP), P(Timeout)]
        pred_probs = self.p.model.predict_proba(feature_vector)[0]
        
        prob_sl = pred_probs[0]  # 觸及停損的機率
        prob_tp = pred_probs[1]  # 觸及停利的機率
        
        current_price = self.data.close[0]

        # --- 做多邏輯 ---
        if prob_tp > prob_sl and prob_tp > self.p.entry_threshold:
            sl_price = current_price * (1 - self.p.sl_pct)
            tp_price = current_price * (1 + self.p.tp_pct)
            self.buy_bracket(
                price=current_price,
                stopprice=sl_price,
                limitprice=tp_price,
            )
            self.log(f"建立買單 @ {current_price:.5f}, TP @ {tp_price:.5f}, SL @ {sl_price:.5f}")

        # --- 做空邏輯 ---
        elif prob_sl > prob_tp and prob_sl > self.p.entry_threshold:
            sl_price = current_price * (1 + self.p.sl_pct)
            tp_price = current_price * (1 - self.p.tp_pct)
            self.sell_bracket(
                price=current_price,
                stopprice=sl_price,
                limitprice=tp_price,
            )
            self.log(f"建立賣單 @ {current_price:.5f}, TP @ {tp_price:.5f}, SL @ {sl_price:.5f}")

# ==============================================================================
# ★★★ 優化器與回測器 (修改版) ★★★
# ==============================================================================
class MLOptimizerAndBacktester:
    def __init__(self, config: Dict):
        # ... 初始化前半部分不變 ...
        self.config = config
        self.paths = config['paths']
        self.wfo_config = config['walk_forward_optimization']
        # ★★★ 策略參數現在直接從三道門檻設定中讀取一部分 ★★★
        self.strategy_params = config['strategy_params']
        self.tb_settings = config['triple_barrier_settings']
        self.strategy_params['tp_pct'] = self.tb_settings['tp_pct']
        self.strategy_params['sl_pct'] = self.tb_settings['sl_pct']
        # ... 後半部分不變 ...
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
        # ... 此函式不變 ...
        if not file_path.exists(): self.logger.error(f"致命錯誤: 輸入檔案 {file_path} 不存在！"); sys.exit(1)
        with open(file_path, 'r') as f: data = json.load(f)
        self.logger.info(f"成功從 {file_path} 載入 {len(data['selected_features'])} 個全域特徵。")
        return data

    # ★★★ 核心修正 #1：修改 objective 函式以支持多分類 ★★★
    def objective(self, trial: optuna.trial.Trial, X_train, y_train, df_val, available_features: list) -> float:
        """Optuna 的目標函數，優化驗證集上的夏普比率 (多分類版)"""
        param = {
            'objective': 'multiclass',    # <--- 修改
            'metric': 'multi_logloss',    # <--- 修改
            'num_class': 3,               # <--- 新增
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'seed': 42,
            'n_jobs': -1,
        }
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train.values, y_train.values)
        
        result = self.run_backtest_on_fold(df_val, model, available_features)
        sharpe = result.get('sharpe_ratio')
        
        if not sharpe or sharpe == 0.0:
            return -1.0
            
        return sharpe

    def run_backtest_on_fold(self, df_fold: pd.DataFrame, model: lgb.LGBMClassifier, available_features: list) -> Dict:
        # ★★★ 此函式只需修改傳遞給策略的參數 ★★★
        class PandasDataWithFeatures(bt.feeds.PandasData):
            lines = tuple(available_features)
            params = (('volume', 'tick_volume'),) + tuple([(f, -1) for f in available_features])

        cerebro = bt.Cerebro(stdstats=False)
        cerebro.adddata(PandasDataWithFeatures(dataname=df_fold))
        
        # 將模型和特徵列表，以及從 config 讀取的策略參數一起傳入
        strategy_kwargs = {
            'model': model,
            'features': available_features,
            **self.strategy_params  # <--- 直接傳遞整個參數字典
        }
        
        cerebro.addstrategy(FinalMLStrategy, **strategy_kwargs)
        cerebro.broker.setcash(self.wfo_config['initial_cash'])
        cerebro.broker.setcommission(commission=self.wfo_config['commission'])
        
        # ... 分析器部分不變 ...
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        
        results = cerebro.run()
        analysis = results[0].analyzers
        trades_analysis = analysis.trades.get_analysis()
        drawdown_analysis = analysis.drawdown.get_analysis()
        sharpe_analysis = analysis.sharpe.get_analysis()
        
        if trades_analysis.get('total', {}).get('total', 0) > 0:
            sharpe_ratio = sharpe_analysis.get('sharperatio')
            return {
                "pnl": trades_analysis.pnl.net.total,
                "total_trades": trades_analysis.total.total,
                "won_trades": trades_analysis.won.total,
                "lost_trades": trades_analysis.lost.total,
                "max_drawdown": drawdown_analysis.max.drawdown,
                "sharpe_ratio": sharpe_ratio if sharpe_ratio is not None else 0.0,
            }
        else:
            return {"pnl": 0.0, "total_trades": 0, "won_trades": 0, "lost_trades": 0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}

    def run_for_single_market(self, market_file_path: Path):
        # ★★★ 此函式需修改標籤生成方式 ★★★
        self.logger.info(f"{'='*25} 開始處理市場: {market_file_path.stem} {'='*25}")
        df = pd.read_parquet(market_file_path)
        df.index = pd.to_datetime(df.index)
        
        available_features = [f for f in self.selected_features if f in df.columns]
        self.logger.info(f"在 {market_file_path.stem} 中找到 {len(available_features)}/{len(self.selected_features)} 個可用特徵。")
        if len(available_features) < 5:
            self.logger.warning(f"可用特徵過少 (<5)，跳過市場 {market_file_path.stem}。")
            return

        # 產生三道門檻標籤
        df = create_triple_barrier_labels(df, self.tb_settings)
        
        # ★★★ 新增：產生多分類目標 ★★★
        # 1: 停利 (TP), -1: 停損 (SL), 0: 時間到期 (Timeout)
        # 我們將其映射為 -> 1: TP, 0: SL, 2: Timeout
        mapping = {1: 1, -1: 0, 0: 2} 
        df['target_multiclass'] = df['label'].map(mapping)

        # 在所有特徵和標籤生成後再刪除 NaN
        df.dropna(inplace=True)

        if df.empty:
            self.logger.warning(f"市場 {market_file_path.stem} 在數據清洗後為空，已跳過。")
            return
            
        # ... 滾動窗口的邏輯不變 ...
        start_date, end_date = df.index.min(), df.index.max()
        train_days, val_days = timedelta(days=self.wfo_config['training_days']), timedelta(days=self.wfo_config['validation_days'])
        test_days, step_days = timedelta(days=self.wfo_config['testing_days']), timedelta(days=self.wfo_config['step_days'])
        current_date, fold_results, fold_number = start_date, [], 0
        
        while current_date + train_days + val_days + test_days <= end_date:
            fold_number += 1
            train_start, val_start = current_date, current_date + train_days
            test_start, test_end = val_start + val_days, val_start + val_days + test_days
            print(f"\n--- Fold {fold_number}: Train[{train_start.date()}-{val_start.date()}] | Val[{val_start.date()}-{test_start.date()}] | Test[{test_start.date()}-{test_end.date()}] ---")
            
            df_train, df_val, df_test = df[train_start:val_start], df[val_start:test_start], df[test_start:test_end]
            
            if any(d.empty for d in [df_train, df_val, df_test]):
                self.logger.warning("當前窗口數據不足，跳過此 Fold。")
                current_date += step_days
                continue
                
            # ★★★ 使用新的多分類目標 'target_multiclass' ★★★
            X_train, y_train = df_train[available_features], df_train['target_multiclass']
            
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, df_val, available_features), n_trials=self.wfo_config['n_trials'], show_progress_bar=True)
            
            self.logger.info(f"參數優化完成！最佳驗證集夏普比率: {study.best_value:.4f}")
            
            # 使用訓練集+驗證集來訓練最終模型
            X_in_sample = pd.concat([df_train[available_features], df_val[available_features]])
            y_in_sample = pd.concat([df_train['target_multiclass'], df_val['target_multiclass']])
            
            # 建立最終模型
            best_params = study.best_params
            best_params.update({'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 3, 'verbosity': -1})
            final_model = lgb.LGBMClassifier(**best_params)
            final_model.fit(X_in_sample.values, y_in_sample.values)
            
            result = self.run_backtest_on_fold(df_test, final_model, available_features)
            fold_results.append(result)
            win_rate = (result['won_trades'] / result['total_trades'] * 100 if result['total_trades'] > 0 else 0)
            print(f"Fold {fold_number} 測試結果: PnL={result['pnl']:.2f}, Trades={result['total_trades']}, WinRate={win_rate:.2f}%")
            
            current_date += step_days
        
        # ... 最終報告部分不變 ...
        if not fold_results: 
            self.logger.warning(f"市場 {market_file_path.stem} 沒有足夠數據完成滾動回測。")
            return
        final_pnl = sum(r['pnl'] for r in fold_results)
        total_trades = sum(r['total_trades'] for r in fold_results)
        won_trades = sum(r['won_trades'] for r in fold_results)
        win_rate = (won_trades / total_trades) if total_trades > 0 else 0.0
        avg_max_drawdown = np.mean([r['max_drawdown'] for r in fold_results])
        valid_sharpes = [r['sharpe_ratio'] for r in fold_results if r['sharpe_ratio'] is not None and r['sharpe_ratio'] != 0.0]
        avg_sharpe_ratio = np.mean(valid_sharpes) if valid_sharpes else 0.0
        print("\n" + f"--- {market_file_path.stem} 滾動優化總結報告 ---")
        print(f"  - 總淨利: {final_pnl:,.2f}")
        print(f"  - 總交易次數: {total_trades}")
        print(f"  - 總勝率: {win_rate:.2%}")
        print(f"  - 平均最大回撤: {avg_max_drawdown:.2f}%")
        print(f"  - 平均夏普比率: {avg_sharpe_ratio:.2f}")
        print("-" * 50)
        self.all_market_results[market_file_path.stem] = {"final_pnl": final_pnl, "total_trades": total_trades, "win_rate": win_rate, "avg_sharpe": avg_sharpe_ratio}

    def run(self):
        # ... 此函式不變 ...
        self.logger.info(f"{'='*25} 整合式滾動優化與回測流程開始 (版本 7.0) {'='*25}")
        input_dir = Path(self.paths['features_data'])
        all_files = list(input_dir.rglob("*.parquet"))
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
            summary_df = pd.DataFrame.from_dict(self.all_market_results, orient='index')
            summary_df.index.name = 'Market'
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
