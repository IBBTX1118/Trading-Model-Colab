# 檔名: 04_ml_model_optimization.py
# 描述: 整合 ML 模型超參數優化與最終樣本外回測的完整流程。
# 版本: 3.8 (輸出修正版：使用 print 確保在 Colab 中顯示報告)

import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import backtrader as bt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import optuna

# ==============================================================================
# 1. 配置區塊
# ==============================================================================
class Config:
    INPUT_FEATURES_FILE = Path("Output_ML_Pipeline/selected_features.json")
    INPUT_DATA_DIR = Path("Output_Feature_Engineering/MarketData_with_All_Features")
    OUTPUT_BASE_DIR = Path("Output_ML_Pipeline")
    TRAIN_VALIDATION_SPLIT_DATE = "2023-01-01"
    OUT_OF_SAMPLE_START_DATE = "2024-01-01"
    LABEL_LOOK_FORWARD_PERIODS: int = 5
    LABEL_RETURN_THRESHOLD: float = 0.005
    N_TRIALS = 100
    INITIAL_CASH = 100000.0
    COMMISSION = 0.001
    ENTRY_PROB_THRESHOLD = 0.50

# ==============================================================================
# 2. Backtrader 最終策略
# ==============================================================================
class FinalMLStrategy(bt.Strategy):
    params = (('model', None), ('features', None), ('entry_threshold', 0.55))

    def __init__(self):
        if not self.p.model or not self.p.features:
            raise ValueError("模型和特徵列表必須提供！")
        self.feature_lines = [getattr(self.data.lines, f) for f in self.p.features]

    def next(self):
        try:
            feature_vector = np.array([line[0] for line in self.feature_lines]).reshape(1, -1)
        except IndexError:
            return
        pred_prob = self.p.model.predict_proba(feature_vector)[0][1]
        if not self.position and pred_prob > self.p.entry_threshold:
            self.buy()
        elif self.position and pred_prob < 0.5:
            self.sell()

# ==============================================================================
# 3. 主流程控制器
# ==============================================================================
class MLOptimizerAndBacktester:
    def __init__(self, config: Config):
        self.config = config
        self.config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        self.selected_features = self._load_json(self.config.INPUT_FEATURES_FILE)['selected_features']
        self.all_backtest_results = {}
        # 移除 logger 設定，因為我們將改用 print
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _load_json(self, file_path: Path) -> Dict:
        if not file_path.exists():
            print(f"致命錯誤: 輸入檔案 {file_path} 不存在！請先執行 03_feature_selection.py。")
            sys.exit(1)
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"資訊: 成功從 {file_path} 載入 {len(data['selected_features'])} 個全域特徵。")
        return data

    def objective(self, trial: optuna.trial.Trial, X_train, y_train, X_val, y_val) -> float:
        param = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
            'boosting_type': 'gbdt', 'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'seed': 42, 'n_jobs': -1,
        }
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train.values, y_train.values)
        return accuracy_score(y_val.values, model.predict(X_val.values))

    def run_final_backtest(self, df: pd.DataFrame, model: lgb.LGBMClassifier, market_name: str):
        class PandasDataWithFeatures(bt.feeds.PandasData):
            lines = tuple(self.selected_features)
            params = (('volume', 'tick_volume'),) + tuple([(f, -1) for f in self.selected_features])
        
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.adddata(PandasDataWithFeatures(dataname=df))
        cerebro.addstrategy(FinalMLStrategy, model=model, features=self.selected_features, entry_threshold=self.config.ENTRY_PROB_THRESHOLD)
        cerebro.broker.setcash(self.config.INITIAL_CASH)
        cerebro.broker.setcommission(commission=self.config.COMMISSION)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        results = cerebro.run()
        analysis = results[0].analyzers
        final_value = cerebro.broker.getvalue()
        pnl = final_value - self.config.INITIAL_CASH
        trades_analysis = analysis.trades.get_analysis()
        sharpe_ratio = analysis.sharpe.get_analysis().get('sharperatio')
        max_drawdown = analysis.drawdown.get_analysis().max.drawdown
        total_trades = trades_analysis.total.total
        win_rate = (trades_analysis.won.total / total_trades) if total_trades > 0 else 0.0
        
        print("\n" + f"--- {market_name} 樣本外最終回測績效報告 ---")
        print(f"  - 最終資產價值: {final_value:,.2f}")
        print(f"  - 總淨利: {pnl:,.2f}")
        print(f"  - 夏普比率: {sharpe_ratio:.2f}" if sharpe_ratio is not None else "  - 夏普比率: N/A (無交易)")
        print(f"  - 最大回撤: {max_drawdown:.2f}%")
        print(f"  - 總交易次數: {total_trades}")
        print(f"  - 勝率: {win_rate:.2%}")
        print("-" * 50)
        
        self.all_backtest_results[market_name] = {
            "final_value": final_value, "pnl": pnl,
            "sharpe_ratio": sharpe_ratio if sharpe_ratio is not None else 0.0,
            "max_drawdown": max_drawdown, "total_trades": total_trades, "win_rate": win_rate
        }

    def run_for_single_market(self, market_file_path: Path):
        market_name = market_file_path.stem
        print(f"\n{'='*25} 開始處理市場: {market_name} {'='*25}")
        df = pd.read_parquet(market_file_path)
        
        current_features = [f for f in self.selected_features if f in df.columns]
        if len(current_features) != len(self.selected_features):
            missing = set(self.selected_features) - set(current_features)
            print(f"警告: 市場 {market_name} 缺少特徵: {missing}，將被忽略。")

        future_returns = df['close'].shift(-self.config.LABEL_LOOK_FORWARD_PERIODS) / df['close'] - 1
        df['target'] = (future_returns > self.config.LABEL_RETURN_THRESHOLD).astype(int)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if df.empty:
            print(f"警告: 市場 {market_name} 在數據清洗後為空，已跳過。")
            return

        df_in_sample = df[df.index < self.config.OUT_OF_SAMPLE_START_DATE]
        df_out_of_sample = df[df.index >= self.config.OUT_OF_SAMPLE_START_DATE]
        if df_in_sample.empty or df_out_of_sample.empty:
            print(f"警告: 市場 {market_name} 的數據不足以進行樣本內外分割，已跳過。")
            return

        df_train = df_in_sample[df_in_sample.index < self.config.TRAIN_VALIDATION_SPLIT_DATE]
        df_val = df_in_sample[df_in_sample.index >= self.config.TRAIN_VALIDATION_SPLIT_DATE]
        if df_train.empty or df_val.empty:
            print(f"警告: 市場 {market_name} 的樣本內數據不足以進行訓練/驗證分割，已跳過。")
            return

        X_train, y_train = df_train[current_features], df_train['target']
        X_val, y_val = df_val[current_features], df_val['target']
        print(f"資訊: 數據分割完成: 訓練({len(X_train)}), 驗證({len(X_val)}), 樣本外({len(df_out_of_sample)})")

        print("資訊: --- 開始執行 LightGBM 超參數優化 ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=self.config.N_TRIALS, show_progress_bar=True)
        print(f"資訊: 超參數優化完成！最佳驗證集準確率: {study.best_value:.4f}")

        print("資訊: --- 使用最佳超參數訓練最終模型 ---")
        X_in_sample, y_in_sample = df_in_sample[current_features], df_in_sample['target']
        final_model = lgb.LGBMClassifier(**study.best_params)
        final_model.fit(X_in_sample.values, y_in_sample.values)
        
        self.run_final_backtest(df_out_of_sample, final_model, market_name)

    def run(self):
        print(f"========= 整合式優化與回測流程開始 (版本 3.8 - Print 輸出) =========")
        input_files = list(self.config.INPUT_DATA_DIR.rglob("*.parquet"))
        if not input_files:
            print(f"致命錯誤: 在 {self.config.INPUT_DATA_DIR} 中找不到任何數據檔案！流程中止。")
            return
        
        print(f"資訊: 找到 {len(input_files)} 個市場檔案，將逐一進行優化與回測。")
        for market_file in input_files:
            try:
                self.run_for_single_market(market_file)
            except Exception as e:
                print(f"錯誤: 處理市場 {market_file.name} 時發生未預期的錯誤: {e}")

        print("\n" + "="*25 + " 所有市場回測結果總結 " + "="*25)
        if self.all_backtest_results:
            summary_df = pd.DataFrame.from_dict(self.all_backtest_results, orient='index')
            summary_df.index.name = 'Market'
            summary_df['final_value'] = summary_df['final_value'].map('{:,.2f}'.format)
            summary_df['pnl'] = summary_df['pnl'].map('{:,.2f}'.format)
            summary_df['sharpe_ratio'] = summary_df['sharpe_ratio'].map('{:.2f}'.format)
            summary_df['max_drawdown'] = summary_df['max_drawdown'].map('{:.2f}%'.format)
            summary_df['win_rate'] = summary_df['win_rate'].map('{:.2%}'.format)
            print("\n" + summary_df.to_string())
        else:
            print("資訊: 沒有任何市場完成回測。")
        print("=" * 70)
        print("========= 所有任務執行完畢，流程結束 =========")

if __name__ == "__main__":
    try:
        config = Config()
        optimizer = MLOptimizerAndBacktester(config)
        optimizer.run()
    except Exception as e:
        print(f"腳本執行時發生未預期的嚴重錯誤: {e}")
        sys.exit(1)
