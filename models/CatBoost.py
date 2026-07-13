"""
CatBoost 薪資預測模型訓練模組。

使用 CatBoostRegressor 搭配 K 折交叉驗證進行超參數搜尋，
以 MAE（絕對誤差）為評估指標，選出最佳參數組合後在全訓練集上重新訓練。

CatBoost 的優勢：
- 原生支援類別特徵（cat_features），不需要額外 encoding
- 對過擬合有較強的抵抗力
- 訓練速度相對穩定

所屬位置：Salary Predictive Analytics Engine/models/CatBoost.py
被以下模組引用：main.py、catboost_main.py
"""

from itertools import product

import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from config import CATBOOST_FIXED_PARAMS, CATBOOST_PARAM_GRID, CV_FOLDS, RANDOM_STATE


def _iter_param_grid(param_grid: dict):
    """
    將超參數搜尋空間展開為所有參數組合的迭代器。

    例如 {'depth': [4, 6], 'lr': [0.05, 0.1]} 會產生：
    {'depth': 4, 'lr': 0.05}、{'depth': 4, 'lr': 0.1}、
    {'depth': 6, 'lr': 0.05}、{'depth': 6, 'lr': 0.1}

    參數：
        param_grid - 超參數名稱到候選值清單的 dict。

    回傳：
        每次 yield 一個完整參數組合 dict 的生成器。
    """
    keys = list(param_grid.keys())
    for values in product(*(param_grid[key] for key in keys)):
        yield dict(zip(keys, values))


def _param_grid_size(param_grid: dict) -> int:
    """
    計算超參數搜尋空間的總組合數。

    用於訓練開始前記錄日誌，讓使用者了解總共要測試幾組參數。

    參數：
        param_grid - 超參數名稱到候選值清單的 dict。

    回傳：
        所有參數的候選值數量之乘積（總組合數）。
    """
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def train_catboost(x_train, y_train, feature_cols: list[str], logger=None):
    """
    以 K 折交叉驗證搜尋最佳超參數，並在全訓練集上重新訓練 CatBoost 模型。

    流程：
    1. 遍歷 CATBOOST_PARAM_GRID 所有參數組合
    2. 每組參數做 CV_FOLDS 折交叉驗證，計算 log 空間的平均 MAE
    3. 選出 CV MAE 最低的參數組合
    4. 以最佳參數在整個訓練集上重新 fit（refit）

    目標欄位 y_train 已套用 log1p，故此處的 MAE 為 log 空間的誤差，
    需搭配 evaluate_predictions 的 expm1 還原才能得到元為單位的誤差。

    參數：
        x_train      - 訓練集特徵 DataFrame（字串類別欄位）。
        y_train      - 訓練集目標 Series（log1p 月薪）。
        feature_cols - 類別特徵欄位名稱清單（傳給 CatBoost Pool 的 cat_features）。
        logger       - 可選的 Logger 物件，用於記錄每組參數的 CV MAE。

    回傳：
        (final_model, best_params, best_mae) 三元組：
        - final_model : 以最佳參數在全訓練集 refit 後的 CatBoostRegressor。
        - best_params : 交叉驗證選出的最佳超參數 dict。
        - best_mae    : 最佳參數的平均 CV MAE（log 空間）。
    """
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_params = None
    best_mae = float("inf")

    if logger:
        logger.info(
            "Model training: CatBoost starts %s-fold CV, parameter sets=%s",
            CV_FOLDS,
            _param_grid_size(CATBOOST_PARAM_GRID),
        )

    for grid_params in _iter_param_grid(CATBOOST_PARAM_GRID):
        # 合併固定參數與當前搜尋參數，加入固定隨機種子確保可重現性。
        params = {
            **CATBOOST_FIXED_PARAMS,
            **grid_params,
            "random_seed": RANDOM_STATE,
        }
        fold_maes = []

        for train_idx, val_idx in kfold.split(x_train):
            x_tr = x_train.iloc[train_idx]
            x_val = x_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            model = CatBoostRegressor(**params)
            # 使用 CatBoost Pool 並指定 cat_features，讓模型原生處理類別特徵。
            model.fit(Pool(x_tr, y_tr, cat_features=feature_cols))
            val_preds = model.predict(x_val)
            fold_maes.append(mean_absolute_error(y_val, val_preds))

        avg_mae = float(np.mean(fold_maes))
        if logger:
            logger.info("Model training: CatBoost params=%s, CV MAE(log)=%.4f", params, avg_mae)
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params

    # 以最佳參數在完整訓練集上 refit，充分利用所有訓練資料。
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(Pool(x_train, y_train, cat_features=feature_cols))
    if logger:
        logger.info("Model training: CatBoost refit completed with best params")
    return final_model, best_params, best_mae
