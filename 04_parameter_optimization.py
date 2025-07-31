# 檔名: 04_ml_model_optimization.py
# 描述: 整合 ML 模型超參數優化與最終樣本外回測的完整流程。
# 版本: 3.6 (修正版：採用單一市場優化與回測邏輯，杜絕數據洩漏)

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
# 1. 配置區塊 (維持不變)
# ==============================================================================
class Config:
    """儲存腳本所需的所有配置參數。"""
    # --- 輸入檔案 ---
    INPUT_FEATURES_FILE = Path("Output_ML_Pipeline/selected_features.json")
    INPUT_DATA_DIR = Path("Output_Feature_Engineering/MarketData_with_All_Features")
    
    # --- 輸出設定 ---
    OUTPUT_BASE_DIR = Path("Output_ML_Pipeline")
    
    # --- 數據分割 ---
    TRAIN_VALIDATION_SPLIT_DATE = "2023-01-01" # 用於分割訓練集和驗證集
    OUT_OF_SAMPLE_START_DATE = "2024-01-01"    # 樣本外測試的起始日期

    # --- Labeling (與 03 保持一致) ---
    LABEL_LOOK_FORWARD_PERIODS: int = 5
    LABEL_RETURN_THRESHOLD: float = 0.005

    # --- Optuna 優化設定 ---
    N_TRIALS = 100 # 優化嘗試次數
    
    # --- 回測設定 ---
    INITIAL_CASH = 100000.0
    COMMISSION = 0.001
    ENTRY_PROB_THRESHOLD = 0.55

    LOG_LEVEL = "INFO"

# ==============================================================================
# 2. Backtrader 最終策略 (維持不變)
# ==============================================================================
class FinalMLStrategy(bt.Strategy):
    params = (
        ('model', None),
        ('features', None),
        ('entry_threshold', 0.55),
    )

    def __init__(self):
        if not self.p.model or not self.p.features:
            raise ValueError("模型和特徵列表必須提供！")
        
        self.feature_lines = []
        for feature_name in self.p.features:
            self.feature_lines.append(getattr(self.data.lines, feature_name))

    def next(self):
        feature_vector = []
        try:
            for line in self.feature_lines:
                feature_vector.append(line[0])
        except IndexError:
            return

        feature_vector = np.array(feature_vector).reshape(1, -1)
        pred_prob = self.p.model.predict_proba(feature_vector)[0][1]

        if not self.position:
            if pred_prob > self.p.entry_threshold:
                self.buy()
        else:
            if pred_prob < 0.5:
                self.sell()

# ==============================================================================
# 3. 主流程控制器 (核心調整部分)
# ==============================================================================
class MLOptimizerAndBacktester:
    # ★★★【調整 1】: __init__ 初始化調整 ★★★
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 不再載入和合併所有數據，而是在迴圈外預先載入全域特徵列表
        self.selected_features = self._load_json(self.config.INPUT_FEATURES_FILE)['selected_features']
        
        # 初始化一個字典來儲存所有市場的最終回測結果，方便比較
        self.all_backtest_results = {}

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.LOG_LEVEL.upper())
        if logger.hasHandlers():
            logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return logger

    def _load_json(self, file_path: Path) -> Dict:
        if not file_path.exists():
            self.logger.critical(f"輸入檔案 {file_path} 不存在！請先執行 03_feature_selection.py。")
            sys.exit(1)
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.logger.info(f"成功從 {file_path} 載入 {len(data['selected_features'])} 個全域特徵。")
        return data
    
    # ★★★【調整 2】: Optuna 的 objective 函式 ★★★
    # 函式簽名改變，現在從外部接收訓練與驗證數據集
    def objective(self, trial: optuna.trial.Trial, X_train, y_train, X_val, y_val) -> float:
        """Optuna 的目標函數，現在專為單一市場的數據進行優化。"""
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'seed': 42,
            'n_jobs': -1,
        }

        model = lgb.LGBMClassifier(**param)
        model.fit(X_train.values, y_train.values)
        
        preds = model.predict(X_val.values)
        accuracy = accuracy_score(y_val.values, preds)
        return accuracy

    # ★★★【調整 3】: 新增 run_final_backtest 函式 ★★★
    # 將回測邏輯獨立出來，方便在迴圈中調用
    def run_final_backtest(self, df: pd.DataFrame, model: lgb.LGBMClassifier, market_name: str):
        """對單一市場的樣本外數據執行回測並打印/儲存報告。"""
        class PandasDataWithFeatures(bt.feeds.PandasData):
            lines = tuple(self.selected_features)
            params = (('volume', 'tick_volume'),) + tuple([(f, -1) for f in self.selected_features])

        data_feed = PandasDataWithFeatures(dataname=df)
        cerebro = bt.Cerebro(stdstats=False) # 關閉預設分析器，只用我們自己添加的
        cerebro.adddata(data_feed)
        cerebro.addstrategy(FinalMLStrategy, model=model, features=self.selected_features,
                            entry_threshold=self.config.ENTRY_PROB_THRESHOLD)
        
        cerebro.broker.setcash(self.config.INITIAL_CASH)
        cerebro.broker.setcommission(commission=self.config.COMMISSION)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        results = cerebro.run()
        strat = results[0]
        analysis = strat.analyzers
        
        # 整理績效報告
        final_value = cerebro.broker.getvalue()
        pnl = final_value - self.config.INITIAL_CASH
        trades_analysis = analysis.trades.get_analysis()
        sharpe_ratio = analysis.sharpe.get_analysis().get('sharperatio', 0.0)
        max_drawdown = analysis.drawdown.get_analysis().max.drawdown
        total_trades = trades_analysis.total.total
        win_rate = (trades_analysis.won.total / total_trades) if total_trades > 0 else 0.0
        
        # 打印報告
        self.logger.info("\n" + f"--- {market_name} 樣本外最終回測績效報告 ---")
        self.logger.info(f"  - 最終資產價值: {final_value:,.2f}")
        self.logger.info(f"  - 總淨利: {pnl:,.2f}")
        self.logger.info(f"  - 夏普比率: {sharpe_ratio:.2f}")
        self.logger.info(f"  - 最大回撤: {max_drawdown:.2f}%")
        self.logger.info(f"  - 總交易次數: {total_trades}")
        self.logger.info(f"  - 勝率: {win_rate:.2%}")
        self.logger.info("-" * 50)
        
        # 儲存報告到字典中
        self.all_backtest_results[market_name] = {
            "final_value": final_value,
            "pnl": pnl,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "win_rate": win_rate
        }

    # ★★★【調整 4】: 新增 run_for_single_market 函式 ★★★
    # 這是針對單一市場執行完整流程的核心函式
    def run_for_single_market(self, market_file_path: Path):
        """針對單一市場檔案，執行從數據準備到回測的完整流程。"""
        market_name = market_file_path.stem
        self.logger.info(f"\n{'='*25} 開始處理市場: {market_name} {'='*25}")

        # 1. 載入並準備【單一市場】數據
        df = pd.read_parquet(market_file_path)
        
        # 檢查所需特徵是否存在
        missing_features = [f for f in self.selected_features if f not in df.columns]
        if missing_features:
            self.logger.warning(f"市場 {market_name} 缺少以下特徵，將被忽略: {missing_features}")
            current_features = [f for f in self.selected_features if f in df.columns]
        else:
            current_features = self.selected_features

        future_returns = df['close'].shift(-self.config.LABEL_LOOK_FORWARD_PERIODS) / df['close'] - 1
        df['target'] = (future_returns > self.config.LABEL_RETURN_THRESHOLD).astype(int)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if df.empty:
            self.logger.warning(f"市場 {market_name} 在數據清洗後為空，已跳過。")
            return

        # 2. 【單一市場】數據分割
        df_in_sample = df[df.index < self.config.OUT_OF_SAMPLE_START_DATE]
        df_out_of_sample = df[df.index >= self.config.OUT_OF_SAMPLE_START_DATE]

        if df_in_sample.empty or df_out_of_sample.empty:
            self.logger.warning(f"市場 {market_name} 的數據不足以進行樣本內外分割，已跳過。")
            return

        df_train = df_in_sample[df_in_sample.index < self.config.TRAIN_VALIDATION_SPLIT_DATE]
        df_val = df_in_sample[df_in_sample.index >= self.config.TRAIN_VALIDATION_SPLIT_DATE]

        if df_train.empty or df_val.empty:
            self.logger.warning(f"市場 {market_name} 的樣本內數據不足以進行訓練/驗證分割，已跳過。")
            return

        X_train, y_train = df_train[current_features], df_train['target']
        X_val, y_val = df_val[current_features], df_val['target']
        
        self.logger.info(f"數據分割完成: 訓練({len(X_train)}), 驗證({len(X_val)}), 樣本外({len(df_out_of_sample)})")

        # 3. 執行 Optuna 超參數優化
        self.logger.info("--- 開始執行 LightGBM 超參數優化 ---")
        study = optuna.create_study(direction='maximize')
        objective_with_data = lambda trial: self.objective(trial, X_train, y_train, X_val, y_val)
        study.optimize(objective_with_data, n_trials=self.config.N_TRIALS, show_progress_bar=True)
        
        self.logger.info(f"超參數優化完成！最佳驗證集準確率: {study.best_value:.4f}")

        # 4. 訓練最終模型
        self.logger.info("--- 使用最佳超參數訓練最終模型 ---")
        X_in_sample, y_in_sample = df_in_sample[current_features], df_in_sample['target']
        final_model = lgb.LGBMClassifier(**study.best_params)
        final_model.fit(X_in_sample.values, y_in_sample.values)
        
        # 5. 執行樣本外最終回測
        self.run_final_backtest(df_out_of_sample, final_model, market_name)


    # ★★★【調整 5】: 全新的主執行函式 run ★★★
    # 舊的 run 函式被替換為一個主迴圈，調用上述的單一市場處理函式

    def run(self):
        """執行完整的主流程：遍歷所有找到的市場數據檔案並逐一處理。"""
        # --- ↓↓↓ 修改成這樣即可 ↓↓↓ ---
        self.logger.info(f"========= 整合式優化與回測流程開始 (版本 3.6 - Per-Market) =========")

        input_files = list(self.config.INPUT_DATA_DIR.rglob("*.parquet"))
        if not input_files:
            self.logger.critical(f"在 {self.config.INPUT_DATA_DIR} 中找不到任何數據檔案！流程中止。")
            sys.exit(1)
        
        self.logger.info(f"找到 {len(input_files)} 個市場檔案，將逐一進行優化與回測。")

        for market_file in input_files:
            try:
                self.run_for_single_market(market_file)
            except Exception as e:
                self.logger.error(f"處理市場 {market_file.name} 時發生未預期的錯誤: {e}", exc_info=True)
        
        # 在所有市場處理完畢後，打印總結報告
        self.logger.info("\n" + "="*25 + " 所有市場回測結果總結 " + "="*25)
        # 將結果轉換為 DataFrame 以便美觀地打印
        if self.all_backtest_results:
            summary_df = pd.DataFrame.from_dict(self.all_backtest_results, orient='index')
            summary_df.index.name = 'Market'
            # 格式化輸出
            summary_df['final_value'] = summary_df['final_value'].map('{:,.2f}'.format)
            summary_df['pnl'] = summary_df['pnl'].map('{:,.2f}'.format)
            summary_df['sharpe_ratio'] = summary_df['sharpe_ratio'].map('{:.2f}'.format)
            summary_df['max_drawdown'] = summary_df['max_drawdown'].map('{:.2f}%'.format)
            summary_df['win_rate'] = summary_df['win_rate'].map('{:.2%}'.format)
            self.logger.info("\n" + summary_df.to_string())
        else:
            self.logger.info("沒有任何市場完成回測。")
        self.logger.info("=" * 70)

        self.logger.info("========= 所有任務執行完畢，流程結束 =========")


if __name__ == "__main__":
    try:
        config = Config()
        optimizer = MLOptimizerAndBacktester(config)
        optimizer.run()
    except Exception as e:
        logging.critical(f"腳本執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
        sys.exit(1)
