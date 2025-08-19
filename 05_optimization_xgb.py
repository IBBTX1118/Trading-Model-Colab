# 檔名: 05_optimization_xgb.py
# 描述: 使用 XGBoost 模型進行參數優化與回測。
# 版本: 13.0 (XGBoost 實驗版)

import sys; import yaml; import json; from pathlib import Path; from typing import Dict
import pandas as pd; import numpy as np; from datetime import timedelta; import traceback; import logging
import backtrader as bt
import xgboost as xgb # ★★★ 導入 xgboost ★★★
import optuna

# ... (除了檔名和版本註解，以及導入 xgboost，輔助函式與策略部分與 04 號腳本完全相同) ...
# ... (此處省略大部分未變動程式碼以求簡潔) ...

class MLOptimizerAndBacktester:
    # ... (__init__, _load_json, _save_json, run_backtest_on_fold 等函式無變動) ...
    
    # ★★★ 核心修改：objective 函數 ★★★
    def objective(self, trial: optuna.trial.Trial, X_train, y_train, df_val, available_features: list, market_name: str) -> float:
        # XGBoost 參數
        model_param = {
            'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': 3,
            'verbosity': 0, 'use_label_encoder': False, 'seed': 42, 'n_jobs': -1,
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),  # L1
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), # L2
        }
        
        # 策略參數 (無變動)
        strategy_param_updates = {
            'entry_threshold': trial.suggest_float('entry_threshold', 0.35, 0.55, step=0.01),
            'tp_atr_multiplier': trial.suggest_float('tp_atr_multiplier', 1.5, 4.0),
            'sl_atr_multiplier': trial.suggest_float('sl_atr_multiplier', 1.0, 2.5),
        }
        if strategy_param_updates['tp_atr_multiplier'] <= strategy_param_updates['sl_atr_multiplier']: return -999.0
        
        # ★★★ 訓練與回測 (改用 XGBClassifier) ★★★
        model = xgb.XGBClassifier(**model_param)
        model.fit(X_train, y_train)
        temp_strategy_params = {**self.strategy_params, **strategy_param_updates}
        result = self.run_backtest_on_fold(df_val, model, available_features, temp_strategy_params)
        
        # 懲罰機制 (無變動)
        min_trades_threshold = 10
        if result.get('total_trades', 0) < min_trades_threshold: return -999.0

        sharpe = result.get('sharpe_ratio', -1.0)
        return sharpe if sharpe is not None else -1.0
    
    # ★★★ 核心修改：run_for_single_market 函數中的最終模型訓練 ★★★
    def run_for_single_market(self, market_file_path: Path):
        # ... (大部分邏輯無變動) ...
        # ... (此處省略大部分未變動程式碼以求簡潔) ...
        X_in_sample = pd.concat([df_train[available_features], df_val[available_features]])
        y_in_sample = pd.concat([df_train['target_multiclass'], df_val['target_multiclass']])
        
        # 取得最佳模型參數
        model_params = {k: v for k, v in study.best_params.items() if k not in self.strategy_params.keys() and k not in ['entry_threshold', 'tp_atr_multiplier', 'sl_atr_multiplier']}
        model_params.update({
            'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': 3,
            'verbosity': 0, 'use_label_encoder': False, 'seed': 42
        })
        
        # ★★★ 使用 XGBClassifier 訓練最終模型 ★★★
        final_model = xgb.XGBClassifier(**model_params)
        final_model.fit(X_in_sample, y_in_sample)
        
        final_test_params = self.strategy_params.copy()
        for k in ['entry_threshold', 'tp_atr_multiplier', 'sl_atr_multiplier']:
            final_test_params[k] = study.best_params.get(k, self.strategy_params[k])
        # ... (後續程式碼無變動) ...
