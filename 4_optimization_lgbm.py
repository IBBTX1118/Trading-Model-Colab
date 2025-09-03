# 檔名: 4_optimization_lgbm.py
# 描述: 【二分類重構版】 - 使用 LightGBM 模型進行參數優化與回測，擴大搜索範圍
# 版本: 15.2 (週期參數化版 - Refactored for Flexibility)

import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import timedelta
import traceback
import logging
import backtrader as bt
import lightgbm as lgb
import optuna
from collections import defaultdict

# ==============================================================================
#                      輔助函式 (與 03 號腳本同步)
# ==============================================================================
def load_config(config_path: str = 'config.yaml') -> Dict:
    """安全載入配置檔案"""
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
    """創建三道門檻標籤（穩定版）"""
    df_out = df.copy()
    tp_multiplier = settings['tp_atr_multiplier']
    sl_multiplier = settings['sl_atr_multiplier']
    max_hold = settings['max_hold_periods']

    # 動態找到ATR欄位
    atr_col_name = None
    for col in df_out.columns:
        if 'D1_ATR_14' in col:
            atr_col_name = col
            break
        elif 'ATR_14' in col:
            atr_col_name = col

    if atr_col_name is None:
        raise ValueError("數據中缺少 ATR 欄位，無法創建標籤。")
    print(f"標籤創建使用ATR欄位: {atr_col_name}")

    outcomes = pd.Series(index=df_out.index, dtype=float, name='label')
    high_series, low_series, atr_series = df_out['high'], df_out['low'], df_out[atr_col_name]

    valid_count = 0
    tp_count = 0
    sl_count = 0
    hold_count = 0

    for i in range(len(df_out) - max_hold):
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

    print(f"標籤統計: 有效={valid_count}, 止盈={tp_count}, 止損={sl_count}, 持有={hold_count}")
    df_out = df_out.join(outcomes.to_frame())
    return df_out

# ==============================================================================
#                      【修改】交易策略 (適應二分類模型)
# ==============================================================================
class BinaryMLStrategy(bt.Strategy):
    """機器學習交易策略 (二分類版)"""

    params = (
        ('model', None),
        ('features', None),
        ('entry_threshold', 0.55),
        ('tp_atr_multiplier', 1.8),
        ('sl_atr_multiplier', 2.0),
        ('risk_per_trade', 0.015),
    )

    def __init__(self):
        if not self.p.model or not self.p.features:
            raise ValueError("模型和特徵列表必須提供！")

        self.trend_indicator = None
        for trend_name in ['D1_is_uptrend', 'is_uptrend']:
            if hasattr(self.data.lines, trend_name):
                self.trend_indicator = getattr(self.data.lines, trend_name)
                print(f"使用趨勢指標: {trend_name}")
                break

        self.atr_indicator = None
        for atr_name in ['D1_ATR_14', 'ATR_14']:
            if hasattr(self.data.lines, atr_name):
                self.atr_indicator = getattr(self.data.lines, atr_name)
                print(f"使用ATR指標: {atr_name}")
                break

        if self.atr_indicator is None: raise ValueError("找不到ATR指標，策略無法運行")
        if self.trend_indicator is None: print("警告: 找不到趨勢指標，將使用默認上漲趨勢")

        self.current_order = None
        self.trade_count = 0

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status == order.Completed:
            if order.isbuy(): self.log(f'買入執行: P={order.executed.price:.5f}, Qty={order.executed.size:.3f}')
            else: self.log(f'賣出執行: P={order.executed.price:.5f}, Qty={order.executed.size:.3f}')
            self.trade_count += 1
        elif order.status in [order.Canceled, order.Margin, order.Rejected]: self.log(f'訂單失敗: {order.getstatusname()}')
        self.current_order = None

    def notify_trade(self, trade):
        if trade.isclosed: self.log(f'交易結束: 淨盈虧={trade.pnlcomm:.2f}')

    def get_feature_values(self) -> Dict:
        try:
            feature_values = {f: getattr(self.data.lines, f)[0] for f in self.p.features if hasattr(self.data.lines, f)}
            if len(feature_values) != len(self.p.features): return None
            for value in feature_values.values():
                if pd.isna(value) or np.isinf(value): return None
            return feature_values
        except Exception as e:
            print(f"獲取特徵值時發生錯誤: {e}")
            return None

    def make_prediction(self, feature_values: Dict) -> float:
        try:
            return self.p.model.predict_proba(pd.DataFrame([feature_values]))[0][1]
        except Exception as e:
            print(f"預測過程發生錯誤: {e}")
            return 0.0

    def calculate_position_size(self, atr_value: float) -> float:
        try:
            portfolio_value = self.broker.getvalue()
            sl_distance = atr_value * self.p.sl_atr_multiplier
            if sl_distance <= 0: return 0
            risk_amount = portfolio_value * self.p.risk_per_trade
            return max(risk_amount / sl_distance, 0.01)
        except Exception as e:
            print(f"計算倉位大小時發生錯誤: {e}")
            return 0.01

    def next(self):
        if self.current_order or self.position: return

        feature_values = self.get_feature_values()
        if feature_values is None: return

        win_prob = self.make_prediction(feature_values)
        current_price = self.data.close[0]
        atr_value = self.atr_indicator[0]
        if atr_value <= 0 or pd.isna(atr_value): return

        is_uptrend = True
        if self.trend_indicator is not None:
            try: is_uptrend = self.trend_indicator[0] > 0.5
            except: pass

        position_size = self.calculate_position_size(atr_value)
        if position_size <= 0: return

        try:
            if is_uptrend and win_prob > self.p.entry_threshold:
                sl_price = current_price - (atr_value * self.p.sl_atr_multiplier)
                tp_price = current_price + (atr_value * self.p.tp_atr_multiplier)
                main_order = self.buy(size=position_size)
                if main_order:
                    self.sell(size=position_size, exectype=bt.Order.Stop, price=sl_price, parent=main_order)
                    self.sell(size=position_size, exectype=bt.Order.Limit, price=tp_price, parent=main_order)
                    self.current_order = main_order
                    self.log(f'買入信號: 勝率={win_prob:.3f}, TP={tp_price:.5f}, SL={sl_price:.5f}')

            elif not is_uptrend and win_prob > self.p.entry_threshold:
                sl_price = current_price + (atr_value * self.p.sl_atr_multiplier)
                tp_price = current_price - (atr_value * self.p.tp_atr_multiplier)
                main_order = self.sell(size=position_size)
                if main_order:
                    self.buy(size=position_size, exectype=bt.Order.Stop, price=sl_price, parent=main_order)
                    self.buy(size=position_size, exectype=bt.Order.Limit, price=tp_price, parent=main_order)
                    self.current_order = main_order
                    self.log(f'賣出信號: 勝率={win_prob:.3f}, TP={tp_price:.5f}, SL={sl_price:.5f}')
        except Exception as e:
            print(f"下單過程發生錯誤: {e}")

# ==============================================================================
#                      優化器與回測器
# ==============================================================================
class MLOptimizerAndBacktester:
    """機器學習優化器與回測器 - 二分類版"""

    def __init__(self, config: Dict):
        self.config = config
        self.paths = config['paths']
        self.wfo_config = config['walk_forward_optimization']
        self.strategy_params = config.get('strategy_params', {})
        self.tb_settings = config['triple_barrier_settings']
        self.strategy_params.update({
            'tp_atr_multiplier': self.tb_settings.get('tp_atr_multiplier', 2.5),
            'sl_atr_multiplier': self.tb_settings.get('sl_atr_multiplier', 1.5),
            'risk_per_trade': self.strategy_params.get('risk_per_trade', 0.02)
        })
        self.logger = self._setup_logger()
        self.output_base_dir = Path(self.paths['ml_pipeline_output'])
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.all_market_results = {}
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _load_json(self, file_path: Path) -> Dict:
        if not file_path.exists():
            self.logger.error(f"檔案不存在: {file_path}")
            return {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception as e:
            self.logger.error(f"載入JSON失敗: {e}")
            return {}

    def _save_json(self, data: Dict, file_path: Path):
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.logger.info(f"成功保存到: {file_path}")
        except Exception as e: self.logger.error(f"保存JSON失敗: {e}")

    def objective(self, trial: optuna.trial.Trial, X_train, y_train, df_val, available_features: List[str]) -> float:
        try:
            model_params = {
                'objective': 'binary', 'metric': 'logloss', 'verbosity': -1,
                'boosting_type': 'gbdt', 'seed': 42, 'n_jobs': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
            }
            neg_count, pos_count = (y_train == 0).sum(), (y_train == 1).sum()
            if pos_count > 0 and neg_count > 0: model_params['scale_pos_weight'] = neg_count / pos_count

            strategy_updates = {
                'entry_threshold': trial.suggest_float('entry_threshold', 0.55, 0.85),
                'tp_atr_multiplier': trial.suggest_float('tp_atr_multiplier', 1.2, 3.0),
                'sl_atr_multiplier': trial.suggest_float('sl_atr_multiplier', 1.5, 3.5),
                'risk_per_trade': trial.suggest_float('risk_per_trade', 0.01, 0.05),
            }
            if strategy_updates['tp_atr_multiplier'] <= strategy_updates['sl_atr_multiplier']: return -999.0

            model = lgb.LGBMClassifier(**model_params)
            model.fit(X_train, y_train)

            df_val_trades_only = df_val[df_val['label'] != 0].copy()
            if df_val_trades_only.empty: return -999.0

            result = self.run_backtest_on_fold(df_val_trades_only, model, available_features, {**self.strategy_params, **strategy_updates})
            if result.get('total_trades', 0) < 10: return -999.0

            sharpe_ratio = result.get('sharpe_ratio', -999.0)
            return sharpe_ratio if not np.isnan(sharpe_ratio) else -999.0
        except Exception as e:
            print(f"優化過程出錯: {e}")
            return -999.0

    def run_backtest_on_fold(self, df_fold: pd.DataFrame, model, available_features: List[str], strategy_params_override: Dict = None) -> Dict:
        if df_fold.empty: return self._get_empty_result()
        try:
            class PandasDataWithFeatures(bt.feeds.PandasData):
                lines = tuple(df_fold.columns)
                params = (('volume', 'tick_volume'),) + tuple([(col, -1) for col in df_fold.columns])

            cerebro = bt.Cerebro(stdstats=False)
            cerebro.adddata(PandasDataWithFeatures(dataname=df_fold))
            
            final_strategy_params = strategy_params_override or self.strategy_params
            cerebro.addstrategy(BinaryMLStrategy, model=model, features=available_features, **final_strategy_params)

            cerebro.broker.setcash(self.wfo_config['initial_cash'])
            cerebro.broker.setcommission(commission=self.wfo_config['commission'])
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

            results = cerebro.run()
            return self._parse_backtest_results(results[0], cerebro) if results else self._get_empty_result()
        except Exception as e:
            print(f"回測執行錯誤: {e}")
            traceback.print_exc()
            return self._get_empty_result()

    def _parse_backtest_results(self, strategy_result, cerebro) -> Dict:
        try:
            final_value = cerebro.broker.getvalue()
            total_pnl = final_value - self.wfo_config['initial_cash']
            trade_analyzer = strategy_result.analyzers.trades.get_analysis()
            total_trades = trade_analyzer.total.total if hasattr(trade_analyzer.total, 'total') else 0
            if total_trades == 0: return self._get_empty_result(pnl=total_pnl)

            won_trades = trade_analyzer.won.total if hasattr(trade_analyzer.won, 'total') else 0
            lost_trades = trade_analyzer.lost.total if hasattr(trade_analyzer.lost, 'total') else 0
            total_won_pnl = trade_analyzer.won.pnl.total if hasattr(trade_analyzer.won, 'pnl') else 0.0
            total_lost_pnl = trade_analyzer.lost.pnl.total if hasattr(trade_analyzer.lost, 'pnl') else 0.0
            max_drawdown = strategy_result.analyzers.drawdown.get_analysis().max.drawdown
            sharpe_ratio = strategy_result.analyzers.sharpe.get_analysis().get('sharperatio', 0.0) or 0.0
            
            return {
                'pnl': total_pnl, 'total_trades': total_trades, 'won_trades': won_trades,
                'lost_trades': lost_trades, 'pnl_won_total': total_won_pnl,
                'pnl_lost_total': total_lost_pnl, 'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio, 'sqn': 0.0,
            }
        except Exception as e:
            print(f"結果解析錯誤: {e}")
            return self._get_empty_result()

    def _get_empty_result(self, pnl=0.0) -> Dict:
        return {'pnl': pnl, 'total_trades': 0, 'won_trades': 0, 'lost_trades': 0,
                'pnl_won_total': 0.0, 'pnl_lost_total': 0.0, 'max_drawdown': 0.0,
                'sharpe_ratio': 0.0, 'sqn': 0.0}

    def run_for_single_market(self, market_file_path: Path, target_tf: str):
        market_name = market_file_path.stem
        self.logger.info(f"\n{'='*20} 開始處理市場: {market_name} {'='*20}")
        try:
            features_filename = self.output_base_dir / f"selected_features_{market_name}.json"
            features_data = self._load_json(features_filename)
            if not features_data:
                self.logger.warning(f"找不到 {market_name} 的特徵檔案，跳過")
                return

            selected_features = features_data['selected_features']
            df = pd.read_parquet(market_file_path)
            df.index = pd.to_datetime(df.index)

            available_features = [f for f in selected_features if f in df.columns]
            if len(available_features) < 5:
                self.logger.warning(f"可用特徵過少 ({len(available_features)})，跳過")
                return
            self.logger.info(f"使用 {len(available_features)}/{len(selected_features)} 個特徵")

            df = create_triple_barrier_labels(df, self.tb_settings)
            df.dropna(subset=available_features + ['label'], inplace=True)
            if df.empty:
                self.logger.warning("清理後數據為空，跳過")
                return

            self._run_walk_forward_optimization(df, available_features, market_name)
        except Exception as e:
            self.logger.error(f"處理 {market_name} 時發生錯誤: {e}")
            traceback.print_exc()

    def _run_walk_forward_optimization(self, df: pd.DataFrame, available_features: List[str], market_name: str):
        start_date, end_date = df.index.min(), df.index.max()
        wfo_days = {k: timedelta(days=self.wfo_config[k]) for k in ['training_days', 'validation_days', 'testing_days', 'step_days']}
        current_date, fold_results, all_fold_best_params = start_date, [], []

        for fold_number in range(1, 100):
            train_start, val_start = current_date, current_date + wfo_days['training_days']
            test_start, test_end = val_start + wfo_days['validation_days'], val_start + wfo_days['validation_days'] + wfo_days['testing_days']
            if test_end > end_date: break

            print(f"\n--- Fold {fold_number}: Train[{train_start.date()}~{val_start.date()}] | Val[{val_start.date()}~{test_start.date()}] | Test[{test_start.date()}~{test_end.date()}] ---")

            try:
                df_train = df.loc[train_start:val_start - timedelta(seconds=1)]
                df_val, df_test = df.loc[val_start:test_start - timedelta(seconds=1)], df.loc[test_start:test_end - timedelta(seconds=1)]

                df_train_trades_only = df_train[df_train['label'] != 0].copy()
                if len(df_train_trades_only) < 50:
                    self.logger.warning(f"Fold {fold_number} 訓練信號不足 ({len(df_train_trades_only)}個)，跳過")
                    current_date += wfo_days['step_days']
                    continue

                df_train_trades_only['target_binary'] = (df_train_trades_only['label'] == 1).astype(int)
                X_train, y_train = df_train_trades_only[available_features], df_train_trades_only['target_binary']

                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: self.objective(trial, X_train, y_train, df_val, available_features),
                               n_trials=self.wfo_config.get('n_trials', 30), show_progress_bar=True)

                self.logger.info(f"優化完成！最佳夏普比率: {study.best_value:.4f}")
                all_fold_best_params.append({'fold': fold_number, 'best_sharpe_in_val': study.best_value, **study.best_params})

                df_in_sample = pd.concat([df_train, df_val])[lambda x: x['label'] != 0].copy()
                df_in_sample['target_binary'] = (df_in_sample['label'] == 1).astype(int)
                X_in_sample, y_in_sample = df_in_sample[available_features], df_in_sample['target_binary']

                model_params = {k: v for k, v in study.best_params.items() if k not in ['entry_threshold', 'tp_atr_multiplier', 'sl_atr_multiplier', 'risk_per_trade']}
                model_params.update({'objective': 'binary', 'metric': 'logloss', 'verbosity': -1, 'seed': 42})
                neg_count, pos_count = (y_in_sample == 0).sum(), (y_in_sample == 1).sum()
                if pos_count > 0 and neg_count > 0: model_params['scale_pos_weight'] = neg_count / pos_count

                final_model = lgb.LGBMClassifier(**model_params).fit(X_in_sample, y_in_sample)

                if not df_test[df_test['label'] != 0].empty:
                    final_test_params = {**self.strategy_params, **{k: study.best_params[k] for k in ['entry_threshold', 'tp_atr_multiplier', 'sl_atr_multiplier', 'risk_per_trade'] if k in study.best_params}}
                    result = self.run_backtest_on_fold(df_test, final_model, available_features, final_test_params)
                    if result:
                        fold_results.append(result)
                        total_trades = result.get('total_trades', 0)
                        win_rate = (result.get('won_trades', 0) / total_trades * 100) if total_trades > 0 else 0.0
                        print(f"Fold {fold_number} 結果: PnL={result.get('pnl', 0):.2f}, 交易={total_trades}, 勝率={win_rate:.1f}%, 夏普={result.get('sharpe_ratio', 0):.3f}")

            except Exception as e: self.logger.error(f"Fold {fold_number} 處理失敗: {e}")
            current_date += wfo_days['step_days']

        self._save_results_and_generate_report(market_name, fold_results, all_fold_best_params)

    def _save_results_and_generate_report(self, market_name: str, fold_results: List[Dict], all_fold_best_params: List[Dict]):
        try:
            params_filename = self.output_base_dir / f"{market_name}_best_params_binary_lgbm.json"
            self._save_json({"market": market_name, "total_folds": len(all_fold_best_params), "model_type": "binary_classification", "folds_data": all_fold_best_params}, params_filename)
            if not fold_results:
                self.logger.warning(f"{market_name} 沒有有效的fold結果")
                return

            final_pnl, total_trades, won_trades = sum(r.get('pnl', 0) for r in fold_results), sum(r.get('total_trades', 0) for r in fold_results), sum(r.get('won_trades', 0) for r in fold_results)
            total_won_pnl, total_lost_pnl = sum(r.get('pnl_won_total', 0) for r in fold_results), sum(r.get('pnl_lost_total', 0) for r in fold_results)
            profit_factor = abs(total_won_pnl / total_lost_pnl) if total_lost_pnl != 0 else float('inf')
            win_rate = (won_trades / total_trades) if total_trades > 0 else 0.0
            avg_max_drawdown = np.mean([r.get('max_drawdown', 0) for r in fold_results])
            valid_sharpes = [r['sharpe_ratio'] for r in fold_results if r.get('sharpe_ratio') and not np.isnan(r['sharpe_ratio'])]
            avg_sharpe_ratio = np.mean(valid_sharpes) if valid_sharpes else 0.0

            report = f"\n{'='*60}\n📊 {market_name} (二分類LightGBM) 滾動優化總結報告\n{'='*60}\n" \
                     f"📈 總淨利: {final_pnl:,.2f}\n🔢 總交易次數: {total_trades}\n🏆 勝率: {win_rate:.2%}\n" \
                     f"💰 獲利因子: {profit_factor:.2f}\n📉 平均最大回撤: {avg_max_drawdown:.2f}%\n" \
                     f"⚡ 平均夏普比率: {avg_sharpe_ratio:.3f}\n🔧 處理的Folds: {len(fold_results)}\n" \
                     f"🎯 模型類型: 二分類 (止盈 vs 止損)\n💾 參數檔案: {params_filename.name}\n{'='*60}"
            print(report)
            self.all_market_results[market_name] = {
                "final_pnl": final_pnl, "total_trades": total_trades, "win_rate": win_rate,
                "profit_factor": profit_factor, "avg_sharpe": avg_sharpe_ratio,
                "avg_drawdown": avg_max_drawdown, "total_folds": len(fold_results),
                "model_type": "binary_classification"
            }
        except Exception as e: self.logger.error(f"保存結果時發生錯誤: {e}")

    def run(self):
        """主運行函數"""
        self.logger.info(f"{'='*50}\n🚀 LightGBM 二分類滾動優化與回測流程開始 (版本 15.2)\n{'='*50}")

        # ★★★ 核心修改：從 config 讀取目標時間週期 ★★★
        target_tf = self.wfo_config.get('target_timeframe', 'H4').upper()
        self.logger.info(f"🎯 鎖定目標時間週期: {target_tf} (來自 config.yaml)")

        input_dir = Path(self.paths['features_data'])
        if not input_dir.exists():
            self.logger.error(f"特徵數據目錄不存在: {input_dir}")
            return

        all_files = list(input_dir.rglob("*.parquet"))
        
        # ★★★ 核心修改：動態篩選檔案 ★★★
        file_suffix = f"_{target_tf}.parquet"
        input_files = [f for f in all_files if file_suffix in f.name]

        self.logger.info(f"📁 找到 {len(input_files)} 個 {target_tf} 市場檔案")
        if not input_files:
            self.logger.error(f"在 {input_dir} 中找不到任何 {target_tf} 數據檔案！")
            return

        for i, market_file in enumerate(sorted(input_files), 1):
            try:
                self.logger.info(f"[{i}/{len(input_files)}] 處理: {market_file.name}")
                self.run_for_single_market(market_file, target_tf)
            except Exception as e:
                self.logger.error(f"處理 {market_file.name} 時發生嚴重錯誤: {e}")
                traceback.print_exc()

        self._generate_final_summary()

    def _generate_final_summary(self):
        print(f"\n{'='*80}\n🎉 所有市場滾動回測最終總結 (二分類LightGBM v15.2)\n{'='*80}")
        if self.all_market_results:
            summary_df = pd.DataFrame.from_dict(self.all_market_results, orient='index')
            summary_df.index.name = 'Market'
            cols_order = ['final_pnl', 'total_trades', 'win_rate', 'profit_factor', 'avg_sharpe', 'avg_drawdown', 'total_folds', 'model_type']
            summary_df = summary_df[[col for col in cols_order if col in summary_df.columns]]
            
            print(f"\n📊 詳細結果:")
            print(summary_df.to_string(float_format="%.4f"))
            
            total_pnl = summary_df['final_pnl'].sum()
            total_trades = summary_df['total_trades'].sum()
            avg_win_rate = summary_df['win_rate'].mean()
            avg_profit_factor = summary_df['profit_factor'].mean()
            avg_sharpe = summary_df['avg_sharpe'].mean()

            print(f"\n📈 總體統計:")
            print(f"   🏦 總盈虧: {total_pnl:,.2f}")
            print(f"   🔢 總交易次數: {total_trades}")
            print(f"   🏆 平均勝率: {avg_win_rate:.2%}")
            print(f"   💰 平均獲利因子: {avg_profit_factor:.2f}")
            print(f"   ⚡ 平均夏普比率: {avg_sharpe:.3f}")
            print(f"   🎯 模型類型: 二分類 (止盈 vs 止損)")
            print(f"\n✅ 盈利市場: {(summary_df['final_pnl'] > 0).sum()}/{len(summary_df)}")
        else:
            self.logger.info("❌ 沒有任何市場完成回測")
        self.logger.info(f"{'='*50} 所有任務執行完畢 {'='*50}")

# ==============================================================================
#                      主程序入口
# ==============================================================================
if __name__ == "__main__":
    try:
        config = load_config()
        if 'walk_forward_optimization' not in config: config['walk_forward_optimization'] = {}
        config['walk_forward_optimization']['n_trials'] = config['walk_forward_optimization'].get('n_trials', 20)
        
        optimizer = MLOptimizerAndBacktester(config)
        optimizer.run()
    except KeyboardInterrupt:
        print("\n⏹️  用戶中斷執行")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 腳本執行時發生嚴重錯誤:")
        traceback.print_exc()
        sys.exit(1)
