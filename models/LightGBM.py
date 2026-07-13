"""
LightGBM 薪資預測模型訓練模組。

使用 LGBMRegressor 搭配 K 折交叉驗證進行超參數搜尋，
以 MAE（絕對誤差）為評估指標，選出最佳參數後在全訓練集重新訓練。

LightGBM 的類別特徵處理方式：
- 欄位需先轉為 pandas Categorical dtype
- 各折之間需用 align_categorical_columns 對齊 categories，
  避免 train fold 與 val fold 的類別集合不一致導致錯誤
- 訓練完成後將 categories 資訊存於模型物件，供 predict_lightgbm 使用

所屬位置：Salary Predictive Analytics Engine/models/LightGBM.py
被以下模組引用：main.py
"""

from itertools import product

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from config import CV_FOLDS, LIGHTGBM_FIXED_PARAMS, LIGHTGBM_PARAM_GRID, RANDOM_STATE
from utils.engineering import align_categorical_columns


def _iter_param_grid(param_grid: dict):
    """
    將超參數搜尋空間展開為所有參數組合的迭代器。

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

    參數：
        param_grid - 超參數名稱到候選值清單的 dict。

    回傳：
        所有參數的候選值數量之乘積（總組合數）。
    """
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def train_lightgbm(x_train, y_train, feature_cols: list[str], logger=None):
    """
    以 K 折交叉驗證搜尋最佳超參數，並在全訓練集上重新訓練 LightGBM 模型。

    流程：
    1. 將所有特徵欄位轉換為 Categorical dtype（LightGBM 原生類別處理）
    2. 遍歷 LIGHTGBM_PARAM_GRID 所有參數組合
    3. 每組參數做 CV_FOLDS 折交叉驗證：
       - 以 align_categorical_columns 對齊 train/val fold 的 categories
       - 計算各折 MAE 後取平均
    4. 選出 CV MAE 最低的參數組合
    5. 以最佳參數在整個訓練集上 refit
    6. 將 categories 資訊存入模型物件（供預測時對齊使用）

    參數：
        x_train      - 訓練集特徵 DataFrame（字串類別欄位）。
        y_train      - 訓練集目標 Series（log1p 月薪）。
        feature_cols - 類別特徵欄位名稱清單。
        logger       - 可選的 Logger 物件。

    回傳：
        (final_model, best_params, best_mae) 三元組：
        - final_model : refit 後的 LGBMRegressor，附有自訂屬性：
                        salary_feature_columns_ 與 salary_categories_。
        - best_params : 最佳超參數 dict。
        - best_mae    : 最佳 CV MAE（log 空間）。
    """
    # 將所有特徵欄位轉為 Categorical，讓 LightGBM 啟用原生類別處理。
    x_train_cat = x_train.copy()
    for column in feature_cols:
        x_train_cat[column] = x_train_cat[column].astype("category")

    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_params = None
    best_mae = float("inf")

    if logger:
        logger.info(
            "Model training: LightGBM starts %s-fold CV, parameter sets=%s",
            CV_FOLDS,
            _param_grid_size(LIGHTGBM_PARAM_GRID),
        )

    for grid_params in _iter_param_grid(LIGHTGBM_PARAM_GRID):
        # 合併固定參數與搜尋參數。
        params = {
            **LIGHTGBM_FIXED_PARAMS,
            **grid_params,
            "random_state": RANDOM_STATE,
        }
        fold_maes = []

        for train_idx, val_idx in kfold.split(x_train_cat):
            x_tr = x_train_cat.iloc[train_idx]
            x_val = x_train_cat.iloc[val_idx]
            # 對齊 train fold 與 val fold 的 Categorical categories，
            # 避免因各折資料不同導致類別集合不一致的錯誤。
            x_tr, x_val = align_categorical_columns(x_tr, x_val, feature_cols)
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            model = LGBMRegressor(**params)
            model.fit(x_tr, y_tr, categorical_feature=feature_cols)
            val_preds = model.predict(x_val)
            fold_maes.append(mean_absolute_error(y_val, val_preds))

        avg_mae = float(np.mean(fold_maes))
        if logger:
            logger.info("Model training: LightGBM params=%s, CV MAE(log)=%.4f", params, avg_mae)
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params

    # 以最佳參數在完整訓練集上 refit。
    final_model = LGBMRegressor(**best_params)
    final_model.fit(x_train_cat, y_train, categorical_feature=feature_cols)

    # 將特徵欄位名稱與各欄位的 categories 存入模型物件，
    # 供 predict_lightgbm 在預測時正確對齊類別。
    final_model.salary_feature_columns_ = list(feature_cols)
    final_model.salary_categories_ = {
        column: x_train_cat[column].cat.categories for column in feature_cols
    }
    if logger:
        logger.info("Model training: LightGBM refit completed with best params")
    return final_model, best_params, best_mae


def predict_lightgbm(model: LGBMRegressor, x_test):
    """
    使用已訓練的 LightGBM 模型對測試集進行預測。

    預測前需將測試集的類別欄位設為與訓練集相同的 categories，
    避免 LightGBM 因類別集合不一致而拋出警告或錯誤。

    參數：
        model  - train_lightgbm 回傳的 LGBMRegressor 物件
                 （需含 salary_feature_columns_ 與 salary_categories_ 屬性）。
        x_test - 測試集特徵 DataFrame（字串類別欄位）。

    回傳：
        預測結果的 ndarray（log1p 月薪）。
    """
    x_test_cat = x_test.copy()
    for column in model.salary_feature_columns_:
        # 先轉為 category，再套用訓練集的 categories（對齊）。
        x_test_cat[column] = x_test_cat[column].astype("category")
        x_test_cat[column] = x_test_cat[column].cat.set_categories(model.salary_categories_[column])
    return model.predict(x_test_cat)
