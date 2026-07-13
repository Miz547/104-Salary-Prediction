"""
XGBoost 薪資預測模型訓練模組。

使用 XGBRegressor 搭配 K 折交叉驗證進行超參數搜尋，
以 MAE（絕對誤差）為評估指標，選出最佳參數後在全訓練集重新訓練。

XGBoost 的類別特徵處理方式：
- XGBoost 不原生支援字串類別，需先以 OrdinalEncoder 轉換為整數
- 每一折的 encoder 分別在 train fold fit，再 transform val fold，
  避免 data leakage
- 最終 refit 時的 encoder 存入模型物件（供 predict_xgboost 使用）

所屬位置：Salary Predictive Analytics Engine/models/XGBoost.py
被以下模組引用：main.py
"""

from itertools import product

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

from config import CV_FOLDS, RANDOM_STATE, XGBOOST_FIXED_PARAMS, XGBOOST_PARAM_GRID


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
        所有參數的候選值數量之乘積。
    """
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def _fit_transform(train_df, other_df=None):
    """
    以 OrdinalEncoder 對類別欄位進行整數編碼。

    Encoder 只在 train_df 上 fit，other_df 僅 transform。
    未見過的類別值（如測試集中出現但訓練集沒有的值）以 -1 表示。

    使用方式：
        - 只有訓練集：_fit_transform(x_tr) → (x_tr_encoded, encoder)
        - 訓練集 + 驗證集：_fit_transform(x_tr, x_val) → (x_tr_enc, x_val_enc, encoder)

    參數：
        train_df - 用於 fit encoder 的訓練 DataFrame。
        other_df - 可選，需要 transform 的另一個 DataFrame（驗證或測試集）。

    回傳：
        other_df 為 None 時：(train_encoded, encoder)
        other_df 不為 None 時：(train_encoded, other_encoded, encoder)
    """
    columns = list(train_df.columns)
    # handle_unknown='use_encoded_value' + unknown_value=-1：
    # 測試集中未見過的類別值以 -1 填充，而非拋出例外。
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_encoded = train_df.copy()
    train_encoded[columns] = encoder.fit_transform(train_encoded[columns].astype(str))
    if other_df is None:
        return train_encoded, encoder

    other_encoded = other_df.copy()
    other_encoded[columns] = encoder.transform(other_encoded[columns].astype(str))
    return train_encoded, other_encoded, encoder


def train_xgboost(x_train, y_train, feature_cols: list[str], logger=None):
    """
    以 K 折交叉驗證搜尋最佳超參數，並在全訓練集上重新訓練 XGBoost 模型。

    流程：
    1. 遍歷 XGBOOST_PARAM_GRID 所有參數組合
    2. 每組參數做 CV_FOLDS 折交叉驗證：
       - 各折分別 fit OrdinalEncoder，避免 data leakage
       - 計算各折 MAE 後取平均
    3. 選出 CV MAE 最低的參數組合
    4. 以最佳參數在整個訓練集上 refit（此時 encoder fit 整個訓練集）
    5. 將 encoder 與欄位名稱存入模型物件（供預測時使用）

    參數：
        x_train      - 訓練集特徵 DataFrame（字串類別欄位）。
        y_train      - 訓練集目標 Series（log1p 月薪）。
        feature_cols - 類別特徵欄位名稱清單（此處用於日誌記錄，XGBoost 不需額外指定）。
        logger       - 可選的 Logger 物件。

    回傳：
        (final_model, best_params, best_mae) 三元組：
        - final_model : refit 後的 XGBRegressor，附有自訂屬性：
                        salary_encoder_ 與 salary_feature_columns_。
        - best_params : 最佳超參數 dict。
        - best_mae    : 最佳 CV MAE（log 空間）。
    """
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_params = None
    best_mae = float("inf")

    if logger:
        logger.info(
            "Model training: XGBoost starts %s-fold CV, parameter sets=%s",
            CV_FOLDS,
            _param_grid_size(XGBOOST_PARAM_GRID),
        )

    for grid_params in _iter_param_grid(XGBOOST_PARAM_GRID):
        # 合併固定參數與搜尋參數。
        params = {
            **XGBOOST_FIXED_PARAMS,
            **grid_params,
            "random_state": RANDOM_STATE,
        }
        fold_maes = []

        for train_idx, val_idx in kfold.split(x_train):
            x_tr = x_train.iloc[train_idx]
            x_val = x_train.iloc[val_idx]
            # 各折獨立 fit encoder，確保驗證集不影響 encoder 的 categories。
            x_tr_encoded, x_val_encoded, _ = _fit_transform(x_tr, x_val)
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            model = XGBRegressor(**params)
            model.fit(x_tr_encoded, y_tr)
            val_preds = model.predict(x_val_encoded)
            fold_maes.append(mean_absolute_error(y_val, val_preds))

        avg_mae = float(np.mean(fold_maes))
        if logger:
            logger.info("Model training: XGBoost params=%s, CV MAE(log)=%.4f", params, avg_mae)
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params

    # 以最佳參數在完整訓練集上 refit，encoder 也以完整訓練集 fit。
    x_train_encoded, encoder = _fit_transform(x_train)
    final_model = XGBRegressor(**best_params)
    final_model.fit(x_train_encoded, y_train)

    # 將 encoder 與欄位名稱存入模型物件，供 predict_xgboost 在預測時使用。
    final_model.salary_encoder_ = encoder
    final_model.salary_feature_columns_ = list(x_train.columns)
    if logger:
        logger.info("Model training: XGBoost refit completed with best params")
    return final_model, best_params, best_mae


def predict_xgboost(model: XGBRegressor, x_test):
    """
    使用已訓練的 XGBoost 模型對測試集進行預測。

    預測前需以訓練時儲存的 OrdinalEncoder 對測試集進行相同的編碼轉換，
    確保特徵的整數映射與訓練時一致。

    參數：
        model  - train_xgboost 回傳的 XGBRegressor 物件
                 （需含 salary_encoder_ 與 salary_feature_columns_ 屬性）。
        x_test - 測試集特徵 DataFrame（字串類別欄位）。

    回傳：
        預測結果的 ndarray（log1p 月薪）。
    """
    x_encoded = x_test.copy()
    columns = model.salary_feature_columns_
    # 使用訓練時儲存的 encoder 對測試集進行 transform（不重新 fit）。
    x_encoded[columns] = model.salary_encoder_.transform(x_encoded[columns].astype(str))
    return model.predict(x_encoded)
