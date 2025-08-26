# 檔名: 04_optimization_lgbm_fixed.py
# 描述: 修復版本 - 解決 KeyError 和 ZeroDivisionError 問題
# 版本: 13.6 (緊急修復版)

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
#                      修復版交易策略 
# ==============================================================================
class FinalMLStrategy(bt.Strategy):
    params = (
        ('model', None), 
        ('features', None), 
        ('entry_threshold', 0.45), 
        ('tp_atr_multiplier', 2.5), 
        ('sl_atr_multiplier', 1.5),
        ('risk_per_trade', 0.01),
    )

    def __init__(self):
        if not self.p.model or not self.p.features: 
            raise ValueError("模型和特徵列表必須提供！")
        
        # 動態獲取趨勢特徵
        d1_uptrend_name = 'D1_is_uptrend' if hasattr(self.data.lines, 'D1_is_uptrend') else 'is_uptrend'
        self.is_uptrend = getattr(self.data.lines, d1_uptrend_name, None)
        
        # 動態獲取ATR特徵
        atr_name = 'D1_ATR_14' if hasattr(self.data.lines, 'D1_ATR_14') else 'ATR_14'
        self.atr = getattr(self.data.lines, atr_name, None)
        
        self.order = None
        self.trade_count = 0

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: 
            return
        if order.status in [order.Completed]:
            if order.isbuy(): 
                print(f'BUY EXECUTED, Price: {order.executed.price:.5f}')
                self.trade_count += 1
            elif order.issell(): 
                print(f'SELL EXECUTED, Price: {order.executed.price:.5f}')
                self.trade_count += 1
        elif order.status in [order.Canceled, order.Margin, order.Rejected]: 
            print('Order Canceled/Margin/Rejected')
        self.order = None

    def next(self):
        # 如果有未完成訂單或持倉，跳過
        if self.order or self.position:
            return
            
        # 檢查數據有效性
        if len(self.data) == 0:
            return
            
        try:
            # 構建特徵向量
            feature_values = []
            for feat in self.p.features:
                if hasattr(self.data.lines, feat):
                    value = getattr(self.data.lines, feat)[0]
                    if pd.isna(value) or np.isinf(value):
                        return  # 如果任何特徵無效，跳過此次交易
                    feature_values.append(value)
                else:
                    print(f"警告：特徵 {feat} 不存在於數據中")
                    return
            
            if len(feature_values) != len(self.p.features):
                return
                
            # 進行預測
            feature_df = pd.DataFrame([feature_values], columns=self.p.features)
            pred_probs = self.p.model.predict_proba(feature_df)[0]
            
            if len(pred_probs) != 3:
                print(f"預測結果維度錯誤: {len(pred_probs)}")
                return
                
            prob_sl, prob_tp, prob_hold = pred_probs[0], pred_probs[1], pred_probs[2]
            
        except Exception as e:
            print(f"預測過程發生錯誤: {e}")
            return

        # 獲取當前價格和ATR
        current_price = self.data.close[0]
        
        if self.atr is None:
            print("警告：ATR 特徵不存在")
            return
            
        atr_value = self.atr[0]
        if atr_value <= 0 or pd.isna(atr_value):
            return

        # 檢查趨勢方向
        market_is_uptrend = True  # 默認值
        if self.is_uptrend is not None:
            trend_value = self.is_uptrend[0]
            market_is_uptrend = trend_value > 0.5
        
        # 計算倉位大小
        portfolio_value = self.broker.getvalue()
        sl_dist_points = atr_value * self.p.sl_atr_multiplier
        
        if sl_dist_points <= 0:
            return
            
        size = (portfolio_value * self.p.risk_per_trade) / sl_dist_points
        
        # 限制最小倉位
        if size < 0.01:
            size = 0.01

        # 交易邏輯
        if market_is_uptrend and prob_tp > prob_sl and prob_tp > self.p.entry_threshold:
            # 看漲交易
            sl_price = current_price - sl_dist_points
            tp_price = current_price + (atr_value * self.p.tp_atr_multiplier)
            
            try:
                main_order = self.buy(size=size, exectype=bt.Order.Market)
                if main_order:
                    self.sell(size=size, exectype=bt.Order.Stop, price=sl_price, parent=main_order)
                    self.sell(size=size, exectype=bt.Order.Limit, price=tp_price, parent=main_order)
                    self.order = main_order
            except Exception as e:
                print(f"買入訂單執行錯誤: {e}")

        elif not market_is_uptrend and prob_sl > prob_tp and prob_sl > self.p.entry_threshold:
            # 看跌交易
            sl_price = current_price + sl_dist_points
            tp_price = current_price - (atr_value * self.p.tp_atr_multiplier)
            
            try:
                main_order = self.sell(size=size, exectype=bt.Order.Market)
                if main_order:
                    self.buy(size=size, exectype=bt.Order.Stop, price=sl_price, parent=main_order)
                    self.buy(size=size, exectype=bt.Order.Limit, price=tp_price, parent=main_order)
                    self.order = main_order
            except Exception as e:
                print(f"賣出訂單執行錯誤: {e}")

# ==============================================================================
#                      修復版優化器與回測器
# ==============================================================================
class MLOptimizerAndBacktester:
    def __init__(self, config: Dict):
        self.config = config; self.paths = config['paths']; self.wfo_config = config['walk_forward_optimization']
        self.strategy_params = config.get('strategy_params', {}); self.tb_settings = config['triple_barrier_settings']
        self.strategy_params['tp_atr_multiplier'] = self.tb_settings.get('tp_atr_multiplier', 2.5)
        self.strategy_params['sl_atr_multiplier'] = self.tb_settings.get('sl_atr_multiplier', 1.5)
        self.strategy_params['risk_per_trade'] = self.strategy_params.get('risk_per_trade', 0.01)
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
        model_param = {
            'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 3, 'verbosity': -1,
            'boosting_type': 'gbdt', 'seed': 42, 'n_jobs': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
        }
        strategy_param_updates = {
            'entry_threshold': trial.suggest_float('entry_threshold', 0.3, 0.5),
            'tp_atr_multiplier': trial.suggest_float('tp_atr_multiplier', 1.5, 3.0),
            'sl_atr_multiplier': trial.suggest_float('sl_atr_multiplier', 0.8, 2.0),
        }
        if strategy_param_updates['tp_atr_multiplier'] <= strategy_param_updates['sl_atr_multiplier']: 
            return -999.0
        
        try:
            model = lgb.LGBMClassifier(**model_param)
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"模型訓練失敗: {e}")
            return -999.0
        
        temp_strategy_params = {**self.strategy_params, **strategy_param_updates}
        result = self.run_backtest_on_fold(df_val, model, available_features, temp_strategy_params)
        
        # 檢查是否有足夠的交易
        if result.get('total_trades', 0) < 5:
            return -999.0

        # 使用盈虧作為優化目標（更直接）
        pnl = result.get('pnl', 0.0)
        return pnl

    def run_backtest_on_fold(self, df_fold, model, available_features, strategy_params_override=None):
        """★★★ 核心修復：完全重寫回測函數，安全處理所有可能的錯誤 ★★★"""
        try:
            # 檢查數據完整性
            if df_fold.empty:
                return self._get_empty_result()
            
            # 準備數據饋送
            all_possible_columns = list(df_fold.columns)
            
            class PandasDataWithFeatures(bt.feeds.PandasData):
                lines = tuple(all_possible_columns)
                params = (('volume', 'tick_volume'),) + tuple([(col, -1) for col in all_possible_columns])

            cerebro = bt.Cerebro(stdstats=False)
            data_feed = PandasDataWithFeatures(dataname=df_fold)
            cerebro.adddata(data_feed)

            # 準備策略參數
            final_strategy_params = strategy_params_override if strategy_params_override is not None else self.strategy_params
            strategy_kwargs = {'model': model, 'features': available_features, **final_strategy_params}
            
            cerebro.addstrategy(FinalMLStrategy, **strategy_kwargs)
            cerebro.broker.setcash(self.wfo_config['initial_cash'])
            cerebro.broker.setcommission(commission=self.wfo_config['commission'])
            
            # 添加分析器
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            
            # 運行回測
            results = cerebro.run()
            
            if not results or len(results) == 0:
                return self._get_empty_result()
            
            strategy_result = results[0]
            
            # 獲取最終資產價值
            final_value = cerebro.broker.getvalue()
            initial_value = self.wfo_config['initial_cash']
            total_pnl = final_value - initial_value
            
            # ★★★ 安全地解析交易分析結果 ★★★
            trade_analyzer = strategy_result.analyzers.trades
            trade_analysis = trade_analyzer.get_analysis()
            
            # 檢查是否有交易發生
            total_trades = 0
            won_trades = 0
            lost_trades = 0
            total_won_pnl = 0.0
            total_lost_pnl = 0.0
            
            if trade_analysis and 'total' in trade_analysis:
                total_info = trade_analysis.get('total', {})
                if isinstance(total_info, dict) and 'total' in total_info:
                    total_trades = total_info['total']
                
                if total_trades > 0:
                    # 安全地獲取勝利交易信息
                    won_info = trade_analysis.get('won', {})
                    if isinstance(won_info, dict):
                        won_trades = won_info.get('total', 0)
                        won_pnl_info = won_info.get('pnl', {})
                        if isinstance(won_pnl_info, dict):
                            total_won_pnl = won_pnl_info.get('total', 0.0)
                    
                    # 安全地獲取失敗交易信息
                    lost_info = trade_analysis.get('lost', {})
                    if isinstance(lost_info, dict):
                        lost_trades = lost_info.get('total', 0)
                        lost_pnl_info = lost_info.get('pnl', {})
                        if isinstance(lost_pnl_info, dict):
                            total_lost_pnl = lost_pnl_info.get('total', 0.0)
            
            # 獲取其他分析器結果
            drawdown_analyzer = strategy_result.analyzers.drawdown
            drawdown_analysis = drawdown_analyzer.get_analysis()
            max_drawdown = 0.0
            if drawdown_analysis and 'max' in drawdown_analysis:
                max_info = drawdown_analysis['max']
                if isinstance(max_info, dict):
                    max_drawdown = max_info.get('drawdown', 0.0)
            
            sharpe_analyzer = strategy_result.analyzers.sharpe
            sharpe_analysis = sharpe_analyzer.get_analysis()
            sharpe_ratio = 0.0
            if sharpe_analysis and 'sharperatio' in sharpe_analysis:
                sharpe_ratio = sharpe_analysis['sharperatio'] or 0.0
            
            return {
                "pnl": total_pnl,
                "total_trades": total_trades,
                "won_trades": won_trades,
                "lost_trades": lost_trades,
                "pnl_won_total": total_won_pnl,
                "pnl_lost_total": total_lost_pnl,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "sqn": 0.0,  # 暫時設為0，避免計算錯誤
            }
        
        except Exception as e:
            print(f"回測執行錯誤: {e}")
            return self._get_empty_result()
    
    def _get_empty_result(self):
        """返回空結果字典"""
        return {
            "pnl": 0.0,
            "total_trades": 0,
            "won_trades": 0,
            "lost_trades": 0,
            "pnl_won_total": 0.0,
            "pnl_lost_total": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sqn": 0.0,
        }

    def run_for_single_market(self, market_file_path: Path):
        market_name = market_file_path.stem
        self.logger.info(f"{'='*25} 開始處理市場: {market_name} {'='*25}")
        
        features_filename = self.output_base_dir / f"selected_features_{market_name}.json"
        features_data = self._load_json(features_filename)
        if not features_data: 
            self.logger.warning(f"找不到市場 {market_name} 的特徵檔案，跳過。"); 
            return
        
        selected_features = features_data['selected_features']
        
        try:
            df = pd.read_parquet(market_file_path)
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            self.logger.error(f"載入數據失敗: {e}")
            return
        
        # 檢查必要欄位
        required_cols = ['open', 'high', 'low', 'close', 'tick_volume'] + selected_features
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.error(f"市場 {market_name} 的數據缺少必要欄位: {missing_cols}，跳過。")
            return

        # 創建標籤
        try:
            df = create_triple_barrier_labels(df, self.tb_settings)
            mapping = {1: 1, -1: 0, 0: 2}
            df['target_multiclass'] = df['label'].map(mapping)
            df.dropna(subset=selected_features + ['target_multiclass', 'label'], inplace=True)
        except Exception as e:
            self.logger.error(f"標籤創建失敗: {e}")
            return
        
        if df.empty: 
            self.logger.warning(f"數據清洗後為空，跳過。"); 
            return
            
        # 滾動優化
        start_date, end_date = df.index.min(), df.index.max()
        wfo_days = {k: timedelta(days=self.wfo_config[k]) for k in ['training_days', 'validation_days', 'testing_days', 'step_days']}
        current_date, fold_results, fold_number = start_date, [], 0
        all_fold_best_params = []
        
        while current_date + wfo_days['training_days'] + wfo_days['validation_days'] + wfo_days['testing_days'] <= end_date:
            fold_number += 1
            train_start = current_date
            val_start = train_start + wfo_days['training_days']
            test_start = val_start + wfo_days['validation_days']
            test_end = test_start + wfo_days['testing_days']
            
            print(f"\n--- Fold {fold_number}: Train[{train_start.date()}-{val_start.date()}] | Val[{val_start.date()}-{test_start.date()}] | Test[{test_start.date()}-{test_end.date()}] ---")
            
            # 分割數據
            df_train_full = df.loc[train_start:val_start-timedelta(seconds=1)]
            df_val = df.loc[val_start:test_start-timedelta(seconds=1)]
            df_test = df.loc[test_start:test_end-timedelta(seconds=1)]
            
            if any(d.empty for d in [df_train_full, df_val, df_test]): 
                self.logger.warning("當前窗口數據不足，跳過。")
                current_date += wfo_days['step_days']
                continue
            
            # 樣本平衡（簡化版）
            trade_signals = df_train_full[df_train_full['label'] != 0]
            hold_signals = df_train_full[df_train_full['label'] == 0]
            
            if not trade_signals.empty and len(hold_signals) > len(trade_signals):
                hold_signals_sampled = hold_signals.sample(n=len(trade_signals), random_state=42)
                df_train = pd.concat([trade_signals, hold_signals_sampled]).sort_index()
                self.logger.info(f"樣本平衡後，訓練集大小: {len(df_train)} (原大小: {len(df_train_full)})")
            else:
                df_train = df_train_full

            X_train, y_train = df_train[selected_features], df_train['target_multiclass']
            
            # Optuna 優化
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, df_val, selected_features, market_name), 
                n_trials=self.wfo_config.get('n_trials', 5), 
                show_progress_bar=True
            )
            
            self.logger.info(f"優化完成！最佳驗證集 PnL: {study.best_value:.4f}")
            self.logger.info(f"最佳參數: {study.best_params}")
            
            params_with_fold = {'fold': fold_number, 'best_pnl_in_val': study.best_value, **study.best_params}
            all_fold_best_params.append(params_with_fold)

            # 訓練最終模型
            X_in_sample = pd.concat([df_train[selected_features], df_val[selected_features]])
            y_in_sample = pd.concat([df_train['target_multiclass'], df_val['target_multiclass']])
            
            model_params = {k: v for k, v in study.best_params.items() if k not in self.strategy_params and k not in ['entry_threshold', 'tp_atr_multiplier', 'sl_atr_multiplier']}
            model_params.update({'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 3, 'verbosity': -1, 'seed': 42})
            
            try:
                final_model = lgb.LGBMClassifier(**model_params)
                final_model.fit(X_in_sample, y_in_sample)
            except Exception as e:
                self.logger.error(f"最終模型訓練失敗: {e}")
                current_date += wfo_days['step_days']
                continue
            
            # 準備測試參數
            final_test_params = self.strategy_params.copy()
            for k in ['entry_threshold', 'tp_atr_multiplier', 'sl_atr_multiplier']:
                if k in study.best_params: 
                    final_test_params[k] = study.best_params[k]
            
            # 測試
            result = self.run_backtest_on_fold(df_test, final_model, selected_features, final_test_params)
            fold_results.append(result)
            
            # ★★★ 修復：安全計算勝率，避免除零錯誤 ★★★
            total_trades = result.get('total_trades', 0)
            won_trades = result.get('won_trades', 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0.0
            
            print(f"Fold {fold_number} 測試結果: PnL={result.get('pnl', 0.0):.2f}, Trades={total_trades}, WinRate={win_rate:.2f}%")
            current_date += wfo_days['step_days']
        
        # 保存參數
        params_filename = self.output_base_dir / f"{market_name}_best_params_lgbm.json"
        self._save_json({"market": market_name, "folds_data": all_fold_best_params}, params_filename)
        
        if not fold_results: 
            self.logger.warning(f"{market_name} 沒有足夠數據完成回測。"); 
            return
            
        # ★★★ 修復：安全計算總結指標 ★★★
        final_pnl = sum(r.get('pnl', 0) for r in fold_results)
        total_trades = sum(r.get('total_trades', 0) for r in fold_results)
        won_trades = sum(r.get('won_trades', 0) for r in fold_results)
        
        total_won_pnl = sum(r.get('pnl_won_total', 0) for r in fold_results)
        total_lost_pnl = sum(
