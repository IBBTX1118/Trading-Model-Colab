# 檔名: 04_optimization_lgbm.py
# 描述: 完整修復版 - 使用 LightGBM 模型進行參數優化與回測
# 版本: 14.0 (完全修復版)

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
#                      輔助函式
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
        raise ValueError(f"數據中缺少 ATR 欄位，無法創建標籤。")
    
    print(f"使用ATR欄位: {atr_col_name}")
    
    # 初始化結果
    outcomes = pd.Series(index=df_out.index, dtype=float, name='label')
    high_series, low_series, atr_series = df_out['high'], df_out['low'], df_out[atr_col_name]
    
    valid_count = 0
    tp_count = 0
    sl_count = 0
    hold_count = 0
    
    for i in range(len(df_out) - max_hold):
        entry_price = df_out['close'].iloc[i]
        atr_at_entry = atr_series.iloc[i]
        
        if atr_at_entry <= 0 or pd.isna(atr_at_entry):
            continue
            
        valid_count += 1
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
    
    print(f"標籤統計: 有效={valid_count}, 止盈={tp_count}, 止損={sl_count}, 持有={hold_count}")
    
    # 合併結果
    df_out = df_out.join(outcomes.to_frame())
    df_out['target'] = (df_out['label'] == 1).astype(int)
    
    return df_out

# ==============================================================================
#                      改進的交易策略
# ==============================================================================
class ImprovedMLStrategy(bt.Strategy):
    """改進的機器學習交易策略"""
    
    params = (
        ('model', None), 
        ('features', None), 
        ('entry_threshold', 0.4), 
        ('tp_atr_multiplier', 2.5), 
        ('sl_atr_multiplier', 1.5),
        ('risk_per_trade', 0.02),
        ('max_position_size', 0.1),
    )

    def __init__(self):
        if not self.p.model or not self.p.features:
            raise ValueError("模型和特徵列表必須提供！")
        
        # 安全獲取趨勢指標
        self.trend_indicator = None
        for trend_name in ['D1_is_uptrend', 'is_uptrend']:
            if hasattr(self.data.lines, trend_name):
                self.trend_indicator = getattr(self.data.lines, trend_name)
                print(f"使用趨勢指標: {trend_name}")
                break
        
        # 安全獲取ATR指標
        self.atr_indicator = None
        for atr_name in ['D1_ATR_14', 'ATR_14']:
            if hasattr(self.data.lines, atr_name):
                self.atr_indicator = getattr(self.data.lines, atr_name)
                print(f"使用ATR指標: {atr_name}")
                break
                
        if self.atr_indicator is None:
            raise ValueError("找不到ATR指標，策略無法運行")
        
        # 交易追蹤
        self.current_order = None
        self.trade_count = 0
        self.last_prediction = None

    def log(self, txt, dt=None):
        """日誌記錄"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')

    def notify_order(self, order):
        """訂單狀態通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status == order.Completed:
            if order.isbuy():
                self.log(f'買入執行: 價格={order.executed.price:.5f}, 數量={order.executed.size:.3f}')
            elif order.issell():
                self.log(f'賣出執行: 價格={order.executed.price:.5f}, 數量={order.executed.size:.3f}')
            self.trade_count += 1
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'訂單失敗: {order.getstatusname()}')
        
        self.current_order = None

    def notify_trade(self, trade):
        """交易結果通知"""
        if trade.isclosed:
            self.log(f'交易結束: 盈虧={trade.pnl:.2f}, 淨盈虧={trade.pnlcomm:.2f}')

    def get_feature_values(self) -> Dict:
        """安全獲取特徵值"""
        try:
            feature_values = {}
            
            for feature in self.p.features:
                if hasattr(self.data.lines, feature):
                    value = getattr(self.data.lines, feature)[0]
                    
                    # 檢查數值有效性
                    if pd.isna(value) or np.isinf(value):
                        return None
                    
                    feature_values[feature] = value
                else:
                    print(f"警告: 特徵 {feature} 不存在")
                    return None
            
            return feature_values
            
        except Exception as e:
            print(f"獲取特徵值時發生錯誤: {e}")
            return None

    def make_prediction(self, feature_values: Dict) -> tuple:
        """進行預測並返回概率"""
        try:
            feature_df = pd.DataFrame([feature_values])
            pred_probs = self.p.model.predict_proba(feature_df)[0]
            
            if len(pred_probs) != 3:
                print(f"預測維度錯誤: 預期3維，實際{len(pred_probs)}維")
                return None, None, None
            
            return pred_probs[0], pred_probs[1], pred_probs[2]  # sl, tp, hold
            
        except Exception as e:
            print(f"預測過程發生錯誤: {e}")
            return None, None, None

    def calculate_position_size(self, atr_value: float) -> float:
        """計算持倉大小"""
        try:
            portfolio_value = self.broker.getvalue()
            
            # 基於ATR的風險計算
            sl_distance = atr_value * self.p.sl_atr_multiplier
            if sl_distance <= 0:
                return 0
            
            # 基於固定風險百分比的倉位計算
            risk_amount = portfolio_value * self.p.risk_per_trade
            position_size = risk_amount / sl_distance
            
            # 限制最大倉位
            max_size = portfolio_value * self.p.max_position_size
            position_size = min(position_size, max_size)
            
            # 最小倉位檢查
            return max(position_size, 0.01)
            
        except Exception as e:
            print(f"計算倉位大小時發生錯誤: {e}")
            return 0.01

    def next(self):
        """主要交易邏輯"""
        # 如果有未完成的訂單或持倉，跳過
        if self.current_order or self.position:
            return
        
        # 檢查數據可用性
        if len(self.data) <= 0:
            return
        
        # 獲取特徵值
        feature_values = self.get_feature_values()
        if feature_values is None:
            return
        
        # 進行預測
        prob_sl, prob_tp, prob_hold = self.make_prediction(feature_values)
        if prob_tp is None:
            return
        
        self.last_prediction = (prob_sl, prob_tp, prob_hold)
        
        # 獲取市場數據
        current_price = self.data.close[0]
        atr_value = self.atr_indicator[0]
        
        if atr_value <= 0 or pd.isna(atr_value):
            return
        
        # 獲取趨勢方向
        is_uptrend = True  # 默認上漲趨勢
        if self.trend_indicator is not None:
            try:
                trend_value = self.trend_indicator[0]
                is_uptrend = trend_value > 0.5
            except:
                pass
        
        # 計算倉位大小
        position_size = self.calculate_position_size(atr_value)
        if position_size <= 0:
            return
        
        # 交易決策邏輯
        try:
            if is_uptrend and prob_tp > prob_sl and prob_tp > self.p.entry_threshold:
                # 看漲信號
                sl_price = current_price - (atr_value * self.p.sl_atr_multiplier)
                tp_price = current_price + (atr_value * self.p.tp_atr_multiplier)
                
                # 下單
                main_order = self.buy(size=position_size)
                if main_order:
                    # 設置止損和止盈
                    self.sell(size=position_size, exectype=bt.Order.Stop, 
                             price=sl_price, parent=main_order)
                    self.sell(size=position_size, exectype=bt.Order.Limit, 
                             price=tp_price, parent=main_order)
                    self.current_order = main_order
                    
            elif not is_uptrend and prob_sl > prob_tp and prob_sl > self.p.entry_threshold:
                # 看跌信號
                sl_price = current_price + (atr_value * self.p.sl_atr_multiplier)
                tp_price = current_price - (atr_value * self.p.tp_atr_multiplier)
                
                # 下單
                main_order = self.sell(size=position_size)
                if main_order:
                    # 設置止損和止盈
                    self.buy(size=position_size, exectype=bt.Order.Stop, 
                            price=sl_price, parent=main_order)
                    self.buy(size=position_size, exectype=bt.Order.Limit, 
                            price=tp_price, parent=main_order)
                    self.current_order = main_order
                    
        except Exception as e:
            print(f"下單過程發生錯誤: {e}")

# ==============================================================================
#                      優化器與回測器
# ==============================================================================
class MLOptimizerAndBacktester:
    """機器學習優化器與回測器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.paths = config['paths']
        self.wfo_config = config['walk_forward_optimization']
        self.strategy_params = config.get('strategy_params', {})
        self.tb_settings = config['triple_barrier_settings']
        
        # 設置策略參數
        self.strategy_params.update({
            'tp_atr_multiplier': self.tb_settings.get('tp_atr_multiplier', 2.5),
            'sl_atr_multiplier': self.tb_settings.get('sl_atr_multiplier', 1.5),
            'risk_per_trade': self.strategy_params.get('risk_per_trade', 0.02)
        })
        
        # 設置日誌
        self.logger = self._setup_logger()
        
        # 創建輸出目錄
        self.output_base_dir = Path(self.paths['ml_pipeline_output'])
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果統計
        self.all_market_results = {}
        
        # 靜音Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _setup_logger(self) -> logging.Logger:
        """設置日誌記錄器"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

    def _load_json(self, file_path: Path) -> Dict:
        """載入JSON檔案"""
        if not file_path.exists():
            self.logger.error(f"檔案不存在: {file_path}")
            return {}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"成功載入特徵檔案: {len(data.get('selected_features', []))} 個特徵")
            return data
        except Exception as e:
            self.logger.error(f"載入JSON檔案失敗: {e}")
            return {}

    def _save_json(self, data: Dict, file_path: Path):
        """保存JSON檔案"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.logger.info(f"成功保存到: {file_path}")
        except Exception as e:
            self.logger.error(f"保存JSON檔案失敗: {e}")

    def objective(self, trial: optuna.trial.Trial, X_train, y_train, df_val, 
                 available_features: List[str], market_name: str) -> float:
        """Optuna優化目標函數"""
        try:
            # LightGBM模型參數
            model_params = {
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'num_class': 3,
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'seed': 42,
                'n_jobs': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
            }
            
            # 策略參數
            strategy_updates = {
                'entry_threshold': trial.suggest_float('entry_threshold', 0.3, 0.6),
                'tp_atr_multiplier': trial.suggest_float('tp_atr_multiplier', 1.5, 4.0),
                'sl_atr_multiplier': trial.suggest_float('sl_atr_multiplier', 0.8, 2.5),
                'risk_per_trade': trial.suggest_float('risk_per_trade', 0.01, 0.05),
            }
            
            # 確保盈虧比合理
            if strategy_updates['tp_atr_multiplier'] <= strategy_updates['sl_atr_multiplier']:
                return -999.0
                
            # 訓練模型
            model = lgb.LGBMClassifier(**model_params)
            model.fit(X_train, y_train)
            
            # 回測評估
            temp_strategy_params = {**self.strategy_params, **strategy_updates}
            result = self.run_backtest_on_fold(df_val, model, available_features, temp_strategy_params)
            
            # 檢查交易數量
            if result.get('total_trades', 0) < 5:
                return -999.0
            
            # 使用總盈虧作為優化目標
            pnl = result.get('pnl', 0.0)
            
            # 加入風險調整
            max_drawdown = result.get('max_drawdown', 0.0)
            if max_drawdown > 20:  # 回撤超過20%，懲罰
                pnl *= 0.5
                
            return pnl
            
        except Exception as e:
            print(f"優化過程出錯: {e}")
            return -999.0

    def run_backtest_on_fold(self, df_fold: pd.DataFrame, model, available_features: List[str], 
                           strategy_params_override: Dict = None) -> Dict:
        """在單個fold上運行回測"""
        try:
            # 驗證數據
            if df_fold.empty:
                return self._get_empty_result()
            
            # 準備數據饋送
            all_columns = list(df_fold.columns)
            
            class PandasDataWithFeatures(bt.feeds.PandasData):
                lines = tuple(all_columns)
                params = (('volume', 'tick_volume'),) + tuple([(col, -1) for col in all_columns])
            
            # 初始化回測引擎
            cerebro = bt.Cerebro(stdstats=False)
            
            # 添加數據
            try:
                data_feed = PandasDataWithFeatures(dataname=df_fold)
                cerebro.adddata(data_feed)
            except Exception as e:
                print(f"數據饋送錯誤: {e}")
                return self._get_empty_result()
            
            # 策略參數
            final_strategy_params = strategy_params_override or self.strategy_params
            strategy_kwargs = {
                'model': model,
                'features': available_features,
                **final_strategy_params
            }
            
            # 添加策略
            cerebro.addstrategy(ImprovedMLStrategy, **strategy_kwargs)
            
            # 設置經紀商
            cerebro.broker.setcash(self.wfo_config['initial_cash'])
            cerebro.broker.setcommission(commission=self.wfo_config['commission'])
            
            # 添加分析器
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            
            # 運行回測
            results = cerebro.run()
            
            if not results:
                return self._get_empty_result()
            
            # 解析結果
            strategy_result = results[0]
            return self._parse_backtest_results(strategy_result, cerebro)
            
        except Exception as e:
            print(f"回測執行錯誤: {e}")
            traceback.print_exc()
            return self._get_empty_result()

    def _parse_backtest_results(self, strategy_result, cerebro) -> Dict:
        """安全解析回測結果"""
        try:
            # 基本盈虧計算
            final_value = cerebro.broker.getvalue()
            initial_value = self.wfo_config['initial_cash']
            total_pnl = final_value - initial_value
            
            # 交易分析
            trade_analyzer = strategy_result.analyzers.trades
            trade_analysis = trade_analyzer.get_analysis()
            
            # 安全提取交易統計
            total_trades = 0
            won_trades = 0
            lost_trades = 0
            total_won_pnl = 0.0
            total_lost_pnl = 0.0
            
            if trade_analysis and 'total' in trade_analysis:
                total_info = trade_analysis['total']
                if isinstance(total_info, dict) and 'total' in total_info:
                    total_trades = total_info['total']
                    
                    if total_trades > 0:
                        # 獲利交易
                        won_info = trade_analysis.get('won', {})
                        if isinstance(won_info, dict):
                            won_trades = won_info.get('total', 0)
                            if 'pnl' in won_info and isinstance(won_info['pnl'], dict):
                                total_won_pnl = won_info['pnl'].get('total', 0.0)
                        
                        # 虧損交易
                        lost_info = trade_analysis.get('lost', {})
                        if isinstance(lost_info, dict):
                            lost_trades = lost_info.get('total', 0)
                            if 'pnl' in lost_info and isinstance(lost_info['pnl'], dict):
                                total_lost_pnl = lost_info['pnl'].get('total', 0.0)
            
            # 回撤分析
            drawdown_analyzer = strategy_result.analyzers.drawdown
            drawdown_analysis = drawdown_analyzer.get_analysis()
            max_drawdown = 0.0
            if drawdown_analysis and 'max' in drawdown_analysis:
                max_info = drawdown_analysis['max']
                if isinstance(max_info, dict):
                    max_drawdown = max_info.get('drawdown', 0.0)
            
            # 夏普比率
            sharpe_analyzer = strategy_result.analyzers.sharpe
            sharpe_analysis = sharpe_analyzer.get_analysis()
            sharpe_ratio = 0.0
            if sharpe_analysis and 'sharperatio' in sharpe_analysis:
                sharpe_ratio = sharpe_analysis['sharperatio']
                if sharpe_ratio is None or np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                    sharpe_ratio = 0.0
            
            return {
                'pnl': total_pnl,
                'total_trades': total_trades,
                'won_trades': won_trades,
                'lost_trades': lost_trades,
                'pnl_won_total': total_won_pnl,
                'pnl_lost_total': total_lost_pnl,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sqn': 0.0,  # 暫時設為0，避免計算錯誤
            }
            
        except Exception as e:
            print(f"結果解析錯誤: {e}")
            return self._get_empty_result()

    def _get_empty_result(self) -> Dict:
        """返回空結果"""
        return {
            'pnl': 0.0,
            'total_trades': 0,
            'won_trades': 0,
            'lost_trades': 0,
            'pnl_won_total': 0.0,
            'pnl_lost_total': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sqn': 0.0,
        }

    def run_for_single_market(self, market_file_path: Path):
        """處理單個市場"""
        market_name = market_file_path.stem
        self.logger.info(f"{'='*30} 開始處理市場: {market_name} {'='*30}")
        
        try:
            # 載入特徵選擇結果
            features_filename = self.output_base_dir / f"selected_features_{market_name}.json"
            features_data = self._load_json(features_filename)
            
            if not features_data:
                self.logger.warning(f"找不到 {market_name} 的特徵檔案，跳過")
                return
                
            selected_features = features_data['selected_features']
            
            # 載入市場數據
            df = pd.read_parquet(market_file_path)
            df.index = pd.to_datetime(df.index)
            
            # 驗證必要欄位
            required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"缺少必要欄位: {missing_cols}")
                return
            
            # 檢查特徵可用性
            available_features = [f for f in selected_features if f in df.columns]
            missing_features = [f for f in selected_features if f not in df.columns]
            
            if missing_features:
                self.logger.warning(f"缺少特徵: {missing_features}")
            
            if len(available_features) < 5:
                self.logger.warning(f"可用特徵過少 ({len(available_features)})，跳過")
                return
                
            self.logger.info(f"使用 {len(available_features)}/{len(selected_features)} 個特徵")
            
            # 創建標籤
            df = create_triple_barrier_labels(df, self.tb_settings)
            mapping = {1: 1, -1: 0, 0: 2}
            df['target_multiclass'] = df['label'].map(mapping)
            
            # 清理數據
            df.dropna(subset=available_features + ['target_multiclass', 'label'], inplace=True)
            
            if df.empty:
                self.logger.warning("清理後數據為空，跳過")
                return
            
            # 執行滾動優化
            self._run_walk_forward_optimization(df, available_features, market_name)
            
        except Exception as e:
            self.logger.error(f"處理 {market_name} 時發生錯誤: {e}")
            traceback.print_exc()

    def _run_walk_forward_optimization(self, df: pd.DataFrame, available_features: List[str], market_name: str):
        """執行滾動窗口優化"""
        start_date, end_date = df.index.min(), df.index.max()
        
        # 計算時間窗口
        wfo_days = {
            k: timedelta(days=self.wfo_config[k]) 
            for k in ['training_days', 'validation_days', 'testing_days', 'step_days']
        }
        
        current_date = start_date
        fold_results = []
        fold_number = 0
        all_fold_best_params = []
        
        total_duration = wfo_days['training_days'] + wfo_days['validation_days'] + wfo_days['testing_days']
        
        while current_date + total_duration <= end_date:
            fold_number += 1
            
            # 計算時間窗口
            train_start = current_date
            val_start = train_start + wfo_days['training_days']
            test_start = val_start + wfo_days['validation_days']
            test_end = test_start + wfo_days['testing_days']
            
            print(f"\n--- Fold {fold_number}: Train[{train_start.date()}~{val_start.date()}] | "
                  f"Val[{val_start.date()}~{test_start.date()}] | Test[{test_start.date()}~{test_end.date()}] ---")
            
            try:
                # 分割數據
                df_train_raw = df.loc[train_start:val_start - timedelta(seconds=1)]
                df_val = df.loc[val_start:test_start - timedelta(seconds=1)]
                df_test = df.loc[test_start:test_end - timedelta(seconds=1)]
                
                if any(d.empty for d in [df_train_raw, df_val, df_test]):
                    self.logger.warning("數據窗口為空，跳過此fold")
                    current_date += wfo_days['step_days']
                    continue
                
                # 樣本平衡
                df_train = self._balance_training_data(df_train_raw)
                
                # 準備訓練數據
                X_train = df_train[available_features]
                y_train = df_train['target_multiclass']
                
                # Optuna優化
                study = optuna.create_study(direction='maximize', study_name=f'{market_name}_fold_{fold_number}')
                study.optimize(
                    lambda trial: self.objective(trial, X_train, y_train, df_val, available_features, market_name),
                    n_trials=self.wfo_config.get('n_trials', 20),
                    show_progress_bar=True
                )
                
                self.logger.info(f"優化完成！最佳PnL: {study.best_value:.2f}")
                self.logger.info(f"最佳參數: {study.best_params}")
                
                # 保存最佳參數
                params_with_fold = {
                    'fold': fold_number,
                    'best_pnl_in_val': study.best_value,
                    **study.best_params
                }
                all_fold_best_params.append(params_with_fold)
                
                # 訓練最終模型並測試
                result = self._train_and_test_final_model(
                    df_train, df_val, df_test, available_features, study.best_params
                )
                
                if result:
                    fold_results.append(result)
                    
                    # 顯示fold結果
                    total_trades = result.get('total_trades', 0)
                    won_trades = result.get('won_trades', 0)
                    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0.0
                    
                    print(f"Fold {fold_number} 結果: PnL={result.get('pnl', 0):.2f}, "
                          f"交易={total_trades}, 勝率={win_rate:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Fold {fold_number} 處理失敗: {e}")
            
            # 移動到下一個窗口
            current_date += wfo_days['step_days']
        
        # 保存結果和生成報告
        self._save_results_and_generate_report(market_name, fold_results, all_fold_best_params)

    def _balance_training_data(self, df_train_raw: pd.DataFrame) -> pd.DataFrame:
        """平衡訓練數據"""
        try:
            trade_signals = df_train_raw[df_train_raw['label'] != 0]
            hold_signals = df_train_raw[df_train_raw['label'] == 0]
            
            if not trade_signals.empty and len(hold_signals) > len(trade_signals):
                # 對持有信號進行下採樣
                hold_signals_sampled = hold_signals.sample(n=len(trade_signals), random_state=42)
                df_train = pd.concat([trade_signals, hold_signals_sampled]).sort_index()
                self.logger.info(f"樣本平衡: {len(df_train)} (原始: {len(df_train_raw)})")
                return df_train
            else:
                return df_train_raw
                
        except Exception as e:
            self.logger.warning(f"樣本平衡失敗，使用原始數據: {e}")
            return df_train_raw

    def _train_and_test_final_model(self, df_train: pd.DataFrame, df_val: pd.DataFrame, 
                                   df_test: pd.DataFrame, available_features: List[str], 
                                   best_params: Dict) -> Dict:
        """訓練最終模型並在測試集上評估"""
        try:
            # 合併訓練和驗證數據
            X_in_sample = pd.concat([df_train[available_features], df_val[available_features]])
            y_in_sample = pd.concat([df_train['target_multiclass'], df_val['target_multiclass']])
            
            # 提取模型參數
            model_params = {
                k: v for k, v in best_params.items() 
                if k not in ['entry_threshold', 'tp_atr_multiplier', 'sl_atr_multiplier', 'risk_per_trade']
            }
            model_params.update({
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'num_class': 3,
                'verbosity': -1,
                'seed': 42
            })
            
            # 訓練最終模型
            final_model = lgb.LGBMClassifier(**model_params)
            final_model.fit(X_in_sample, y_in_sample)
            
            # 準備測試參數
            final_test_params = self.strategy_params.copy()
            for k in ['entry_threshold', 'tp_atr_multiplier', 'sl_atr_multiplier', 'risk_per_trade']:
                if k in best_params:
                    final_test_params[k] = best_params[k]
            
            # 在測試集上評估
            result = self.run_backtest_on_fold(df_test, final_model, available_features, final_test_params)
            return result
            
        except Exception as e:
            self.logger.error(f"最終模型訓練/測試失敗: {e}")
            return None

    def _save_results_and_generate_report(self, market_name: str, fold_results: List[Dict], 
                                        all_fold_best_params: List[Dict]):
        """保存結果並生成報告"""
        try:
            # 保存參數
            params_filename = self.output_base_dir / f"{market_name}_best_params_lgbm.json"
            self._save_json({
                "market": market_name,
                "total_folds": len(all_fold_best_params),
                "folds_data": all_fold_best_params
            }, params_filename)
            
            if not fold_results:
                self.logger.warning(f"{market_name} 沒有有效的fold結果")
                return
            
            # 計算總體統計
            final_pnl = sum(r.get('pnl', 0) for r in fold_results)
            total_trades = sum(r.get('total_trades', 0) for r in fold_results)
            won_trades = sum(r.get('won_trades', 0) for r in fold_results)
            
            total_won_pnl = sum(r.get('pnl_won_total', 0) for r in fold_results)
            total_lost_pnl = sum(r.get('pnl_lost_total', 0) for r in fold_results)
            
            # 計算衍生指標
            profit_factor = abs(total_won_pnl / total_lost_pnl) if total_lost_pnl != 0 else float('inf')
            win_rate = (won_trades / total_trades) if total_trades > 0 else 0.0
            avg_max_drawdown = np.mean([r.get('max_drawdown', 0) for r in fold_results])
            
            valid_sharpes = [r['sharpe_ratio'] for r in fold_results 
                           if r.get('sharpe_ratio') is not None and not np.isnan(r['sharpe_ratio'])]
            avg_sharpe_ratio = np.mean(valid_sharpes) if valid_sharpes else 0.0
            
            # 生成報告
            report = (
                f"\n{'='*60}\n"
                f"📊 {market_name} (LightGBM) 滾動優化總結報告\n"
                f"{'='*60}\n"
                f"📈 總淨利: {final_pnl:,.2f}\n"
                f"🔢 總交易次數: {total_trades}\n"
                f"🏆 勝率: {win_rate:.2%}\n"
                f"💰 獲利因子: {profit_factor:.2f}\n"
                f"📉 平均最大回撤: {avg_max_drawdown:.2f}%\n"
                f"⚡ 平均夏普比率: {avg_sharpe_ratio:.2f}\n"
                f"🔧 處理的Folds: {len(fold_results)}\n"
                f"💾 參數檔案: {params_filename.name}\n"
                f"{'='*60}"
            )
            
            print(report)
            
            # 保存到總體結果
            self.all_market_results[market_name] = {
                "final_pnl": final_pnl,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_sharpe": avg_sharpe_ratio,
                "avg_drawdown": avg_max_drawdown,
                "total_folds": len(fold_results)
            }
            
        except Exception as e:
            self.logger.error(f"保存結果時發生錯誤: {e}")

    def run(self):
        """主運行函數"""
        self.logger.info(f"{'='*50}")
        self.logger.info(f"🚀 LightGBM 整合式滾動優化與回測流程開始 (版本 14.0)")
        self.logger.info(f"{'='*50}")
        
        # 查找輸入檔案
        input_dir = Path(self.paths['features_data'])
        if not input_dir.exists():
            self.logger.error(f"特徵數據目錄不存在: {input_dir}")
            return
            
        all_files = list(input_dir.rglob("*.parquet"))
        input_files = [f for f in all_files if '_H4.parquet' in f.name]
        
        self.logger.info(f"📁 找到 {len(input_files)} 個 H4 市場檔案")
        
        if not input_files:
            self.logger.error(f"在 {input_dir} 中找不到任何 H4 數據檔案！")
            return
        
        # 處理每個市場
        for i, market_file in enumerate(sorted(input_files), 1):
            try:
                self.logger.info(f"[{i}/{len(input_files)}] 處理: {market_file.name}")
                self.run_for_single_market(market_file)
            except Exception as e:
                self.logger.error(f"處理 {market_file.name} 時發生嚴重錯誤: {e}")
                traceback.print_exc()
        
        # 生成最終總結
        self._generate_final_summary()

    def _generate_final_summary(self):
        """生成最終總結報告"""
        print(f"\n{'='*80}")
        print(f"🎉 所有市場滾動回測最終總結 (LightGBM v14.0)")
        print(f"{'='*80}")
        
        if self.all_market_results:
            # 創建總結DataFrame
            summary_df = pd.DataFrame.from_dict(self.all_market_results, orient='index')
            summary_df.index.name = 'Market'
            
            # 排序列
            cols_order = ['final_pnl', 'total_trades', 'win_rate', 'profit_factor', 
                         'avg_sharpe', 'avg_drawdown', 'total_folds']
            available_cols = [col for col in cols_order if col in summary_df.columns]
            summary_df = summary_df[available_cols]
            
            # 顯示結果
            print(f"\n📊 詳細結果:")
            print(summary_df.to_string(float_format="%.4f"))
            
            # 計算總體統計
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
            print(f"   ⚡ 平均夏普比率: {avg_sharpe:.2f}")
            
            # 性能評估
            profitable_markets = (summary_df['final_pnl'] > 0).sum()
            print(f"\n✅ 盈利市場: {profitable_markets}/{len(summary_df)}")
            
        else:
            self.logger.info("❌ 沒有任何市場完成回測")
            
        self.logger.info(f"{'='*50} 所有任務執行完畢 {'='*50}")

# ==============================================================================
#                      主程序入口
# ==============================================================================
if __name__ == "__main__":
    try:
        # 載入配置
        config = load_config()
        
        # 為快速測試設置較少的試驗次數
        if 'walk_forward_optimization' not in config:
            config['walk_forward_optimization'] = {}
        config['walk_forward_optimization']['n_trials'] = 10
        
        # 創建並運行優化器
        optimizer = MLOptimizerAndBacktester(config)
        optimizer.run()
        
    except KeyboardInterrupt:
        print("\n用戶中斷執行")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 腳本執行時發生嚴重錯誤:")
        traceback.print_exc()
        sys.exit(1)
