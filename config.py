"""
全專案共用設定模組。

此模組集中管理所有訓練超參數、路徑設定與模型設定，
是整個薪資預測專案的「單一真相來源」。
修改此模組的值即可改變全部訓練流程的行為，
不需要逐一修改各個訓練腳本。

所屬位置：Salary Predictive Analytics Engine/config.py
被以下模組引用：main.py、catboost_main.py、models/、utils/
"""

from pathlib import Path

# =========================
# 全專案共用設定
# =========================

# 隨機數種子：控制 train/test split、KFold、模型訓練的可重現性。
RANDOM_STATE = 42

# 原始資料檔案路徑。
DATA_FILE = Path("data/104_0605.csv")

# 清洗完成、且包含特徵工程欄位的資料輸出位置。
CLEAN_DATA_DIR = Path("data/clean data")

# 模型訓練輸出資料夾。
OUTPUT_DIR = Path("outputs")
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
LOG_DIR = OUTPUT_DIR / "logs"
LOG_FILE = LOG_DIR / "training.log"

# 目標欄位名稱：薪資清洗後會產生這個欄位作為模型預測目標。
TARGET_COLUMN = "target_monthly_salary"

# 薪資目標類型：max 代表取薪資範圍最大值，min 代表取薪資範圍最小值。
SALARY_BOUNDS = ["max", "min"]

# 測試集比例：0.2 代表 80% 訓練、20% 測試。
TEST_SIZE = 0.2

# 文字特徵工程設定。
# TEXT_CLUSTER_COUNT：KMeans 將職缺文字分成幾群。
# TFIDF_MAX_FEATURES：TF-IDF 最多保留多少文字特徵。
TEXT_CLUSTER_COUNT = 10
TFIDF_MAX_FEATURES = 4000

# 交叉驗證折數：三個模型都會使用這個折數搜尋最佳參數。
CV_FOLDS = 3

# 預測誤差門檻：用來計算「預測誤差 <= 25% 比例」。
ERROR_THRESHOLD_PERCENT = 25


# =========================
# CatBoost 訓練參數
# =========================

# CATBOOST_PARAM_GRID：
# 訓練工程師調參時主要改這裡。每個 list 裡可以放多個值，
# 程式會用交叉驗證測試所有組合，選出 MAE(log) 最低的參數。
CATBOOST_PARAM_GRID = {
    "depth": [6],
    "learning_rate": [0.1],
    "iterations": [120],
}

# CATBOOST_FIXED_PARAMS：
# 每次 CatBoost 訓練都固定使用的參數。通常不需要頻繁調整。
CATBOOST_FIXED_PARAMS = {
    "loss_function": "MAE",
    "thread_count": 1,
    "allow_writing_files": False,
    "verbose": 0,
}


# =========================
# LightGBM 訓練參數
# =========================

# LIGHTGBM_PARAM_GRID：
# LightGBM 的交叉驗證搜尋空間。增加數值會讓搜尋更完整，但訓練時間也會增加。
LIGHTGBM_PARAM_GRID = {
    "num_leaves": [31, 63],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [150, 250],
    "max_depth": [-1, 8],
}

# LIGHTGBM_FIXED_PARAMS：
# LightGBM 固定參數。objective=regression_l1 代表用 MAE 方向訓練回歸模型。
LIGHTGBM_FIXED_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "verbose": -1,
}


# =========================
# XGBoost 訓練參數
# =========================

# XGBOOST_PARAM_GRID：
# XGBoost 的交叉驗證搜尋空間。常調整 max_depth、learning_rate、n_estimators。
XGBOOST_PARAM_GRID = {
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200],
    "subsample": [1.0],
    "colsample_bytree": [0.8],
}

# XGBOOST_FIXED_PARAMS：
# XGBoost 固定參數。objective=reg:absoluteerror 代表以絕對誤差方向訓練。
XGBOOST_FIXED_PARAMS = {
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "enable_categorical": True,
    "verbosity": 0,
}
