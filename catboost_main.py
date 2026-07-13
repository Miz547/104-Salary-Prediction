"""
CatBoost 單獨訓練入口（支援命令列參數）。

此腳本專為「變數模擬器」（catboost_training.html）設計，
允許前端透過 salary_server.py 觸發，以自訂的目標薪資與特徵欄位
重新訓練 CatBoost 模型，而不影響其他兩個模型（LightGBM / XGBoost）。

命令列用法：
    # 訓練 max 薪資模型（使用全部預設特徵）
    python catboost_main.py --target max

    # 訓練 min 薪資模型，只使用指定特徵欄位
    python catboost_main.py --target min --features keyword,clean_location,experience_band

    # 訓練 max + min（不指定 --target 則訓練兩者）
    python catboost_main.py

所屬位置：Salary Predictive Analytics Engine/catboost_main.py
被以下模組引用：salary_server.py（透過 subprocess 呼叫）
"""

from pathlib import Path
import argparse
import os

# 限制 joblib 平行處理的 CPU 核心數為 1，避免 Windows 子行程問題。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd

from config import CLEAN_DATA_DIR, DATA_FILE, LOG_FILE, SALARY_BOUNDS, VISUALIZATION_DIR
from models.CatBoost import train_catboost
from utils.cleaning import read_104_csv
from utils.engineering import build_model_frame, split_data
from utils.logger import setup_logger
from utils.visualization import evaluate_predictions, save_visualizations


# 不能作為訓練特徵的欄位：目標欄位本身及其 log 轉換版本。
EXCLUDED_FEATURE_COLUMNS = {"target_monthly_salary", "target_log1p"}

# 此腳本只訓練 CatBoost，不包含 LightGBM / XGBoost。
MODEL_RUNNERS = {
    "CatBoost": {
        "train": train_catboost,
        "predict": lambda model, x_test: model.predict(x_test),
    },
}


def parse_feature_list(value: str | None) -> list[str] | None:
    """
    解析命令列傳入的逗號分隔特徵欄位字串。

    例如：'keyword,clean_location,experience_band'
          → ['keyword', 'clean_location', 'experience_band']

    空字串或 None 會回傳 None，代表使用全部預設特徵。

    參數：
        value - 逗號分隔的特徵欄位字串，或 None。

    回傳：
        特徵欄位名稱清單，或 None（使用預設特徵）。
    """
    if not value:
        return None
    features = [item.strip() for item in value.split(",") if item.strip()]
    return features or None


def select_training_features(
    df_model: pd.DataFrame,
    default_x: pd.DataFrame,
    default_features: list[str],
    selected_features: list[str] | None,
    logger=None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    依前端傳入的特徵欄位清單篩選訓練特徵。

    若 selected_features 為 None，直接回傳 default_x 與 default_features。
    若有指定，則：
    1. 過濾掉不存在於 df_model 的欄位（記錄 warning）
    2. 排除目標欄位（EXCLUDED_FEATURE_COLUMNS）
    3. 若無任何有效特徵，拋出 ValueError

    參數：
        df_model          - build_model_frame 回傳的完整訓練 DataFrame。
        default_x         - 預設的特徵 DataFrame（全部特徵）。
        default_features  - 預設的特徵欄位名稱清單。
        selected_features - 前端指定的特徵欄位名稱清單，或 None。
        logger            - 可選的 Logger 物件。

    回傳：
        (x, feature_cols) 元組：篩選後的特徵 DataFrame 與欄位名稱清單。
    """
    if not selected_features:
        # 未指定特徵，使用全部預設特徵。
        return default_x, default_features

    # 只保留實際存在於 df_model 且不是目標欄位的特徵。
    valid_features = [
        feature
        for feature in selected_features
        if feature in df_model.columns and feature not in EXCLUDED_FEATURE_COLUMNS
    ]
    # 記錄不存在的欄位名稱，方便除錯。
    missing_features = [feature for feature in selected_features if feature not in df_model.columns]

    if logger and missing_features:
        logger.warning("Skipped unavailable selected X features: %s", missing_features)
    if not valid_features:
        raise ValueError("No valid selected X features were found in the training dataframe")

    x = df_model[valid_features].copy()
    # 確保所有特徵欄位為字串型別（CatBoost cat_features 需求）。
    for column in valid_features:
        x[column] = x[column].astype(str)
    if logger:
        logger.info("Using selected X features from frontend: %s", valid_features)
    return x, valid_features


def format_result_message(run_name: str, best_params: dict, cv_mae_log: float, metrics: dict) -> str:
    """
    格式化訓練結果摘要訊息，用於日誌輸出。

    參數：
        run_name    - 模型執行名稱，如 'CatBoost_max'。
        best_params - 交叉驗證選出的最佳超參數 dict。
        cv_mae_log  - 最佳 CV MAE（log 空間）。
        metrics     - evaluate_predictions 回傳的評估指標 dict。

    回傳：
        多行格式化字串，包含 MAE、RMSE、R2、最佳參數等訊息。
    """
    return (
        f"{run_name} training completed\n"
        f"1. MAE={metrics['mae']:,.2f}, RMSE={metrics['rmse']:,.2f}, R2={metrics['r2']:.4f}\n"
        f"2. Best params={best_params}\n"
        f"3. CV MAE(log)={cv_mae_log:.4f}\n"
        f"4. MAPE={metrics['mape']:.2f}%\n"
        f"5. Error <= 25%={metrics['within_threshold_percent']:.2f}%"
    )


def save_clean_feature_data(df_model: pd.DataFrame, y: pd.Series, salary_bound: str, logger=None) -> Path:
    """
    將清洗後的特徵資料儲存為 CSV，供後續分析或除錯使用。

    輸出位置：data/clean data/104_salary_clean_features_{salary_bound}.csv

    參數：
        df_model     - build_model_frame 回傳的完整訓練 DataFrame。
        y            - 目標 Series（log1p 月薪）。
        salary_bound - 'max' 或 'min'，用於區分輸出檔名。
        logger       - 可選的 Logger 物件。

    回傳：
        儲存完成的 CSV 檔案路徑。
    """
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    export_df = df_model.copy()
    export_df["target_log1p"] = y.to_numpy()
    output_path = CLEAN_DATA_DIR / f"104_salary_clean_features_{salary_bound}.csv"
    export_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    if logger:
        logger.info(
            "Clean data export: saved cleaned feature data for salary_%s to %s, rows=%s, columns=%s",
            salary_bound,
            output_path.resolve(),
            export_df.shape[0],
            export_df.shape[1],
        )
    return output_path


def run_training_pipeline(
    data_file: Path = DATA_FILE,
    selected_targets: list[str] | None = None,
    selected_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    執行 CatBoost 訓練流程（支援自訂目標與特徵）。

    流程：
    1. 讀取原始 CSV 資料
    2. 對指定的薪資目標（max / min 或兩者）分別：
       a. 建立特徵框架（清洗、聚類、衍生特徵）
       b. 依 selected_features 篩選訓練特徵
       c. 儲存清洗後特徵 CSV
       d. 切分訓練/測試集
       e. 訓練 CatBoost 模型
       f. 評估並產生視覺化報告
    3. 將訓練結果彙整為 training_summary.csv

    參數：
        data_file         - 原始 CSV 資料路徑，預設使用 config.DATA_FILE。
        selected_targets  - 指定要訓練的薪資目標清單（['max'] / ['min'] / None=兩者）。
        selected_features - 指定要使用的特徵欄位清單（None=使用全部預設特徵）。

    回傳：
        包含訓練結果的 DataFrame（同時儲存為 training_summary.csv）。
    """
    logger = setup_logger(log_file=LOG_FILE)
    logger.info("Starting CatBoost salary training")
    logger.info("Config: data_file=%s, visualization_dir=%s, log_file=%s", data_file, VISUALIZATION_DIR, LOG_FILE)

    # 若未指定目標，則訓練全部薪資目標（max + min）。
    targets = selected_targets or SALARY_BOUNDS
    # 驗證目標值是否合法。
    invalid_targets = [target for target in targets if target not in SALARY_BOUNDS]
    if invalid_targets:
        raise ValueError(f"Unsupported target values: {invalid_targets}")

    df = read_104_csv(data_file, logger=logger)
    results = []

    for salary_bound in targets:
        logger.info("========== Training target: %s ==========", salary_bound)
        x, y, feature_cols, token_counter, df_model = build_model_frame(df, salary_bound, logger=logger)
        # 依前端傳入的特徵欄位清單篩選（或使用全部預設特徵）。
        x, feature_cols = select_training_features(
            df_model=df_model,
            default_x=x,
            default_features=feature_cols,
            selected_features=selected_features,
            logger=logger,
        )
        save_clean_feature_data(df_model, y, salary_bound, logger=logger)
        x_train, x_test, y_train, y_test = split_data(x, y, logger=logger)

        for model_name, runner in MODEL_RUNNERS.items():
            run_name = f"{model_name}_{salary_bound}"
            logger.info("---------- Training %s ----------", run_name)
            model, best_params, cv_mae_log = runner["train"](x_train, y_train, feature_cols, logger=logger)
            logger.info("Model fitted: %s", run_name)
            y_pred_log = runner["predict"](model, x_test)
            metrics = evaluate_predictions(y_test, y_pred_log)

            logger.info("Saving report and visualizations for %s", run_name)
            report_path = save_visualizations(
                model=model,
                df_model=df_model,
                feature_cols=feature_cols,
                token_counter=token_counter,
                metrics=metrics,
                best_params=best_params,
                output_dir=VISUALIZATION_DIR,
                prefix=run_name.lower(),
            )

            result = {
                "target": salary_bound,
                "model": model_name,
                "features": ",".join(feature_cols),
                "cv_mae_log": cv_mae_log,
                "test_mae": metrics["mae"],
                "test_rmse": metrics["rmse"],
                "test_r2": metrics["r2"],
                "test_mape": metrics["mape"],
                "within_25_percent": metrics["within_threshold_percent"],
                "best_params": str(best_params),
                "report": str(report_path),
            }
            results.append(result)
            logger.info("\n%s", format_result_message(run_name, best_params, cv_mae_log, metrics))
            logger.info("Report saved: %s", report_path.resolve())

    results_df = pd.DataFrame(results)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = VISUALIZATION_DIR / "training_summary.csv"
    results_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info("Training summary saved: %s", summary_path.resolve())
    logger.info("Visualization directory: %s", VISUALIZATION_DIR.resolve())
    logger.info("Training finished")
    return results_df


def main() -> None:
    """
    命令列入口：解析 --target 與 --features 參數後執行訓練流程。

    --target  : 'max' 或 'min'，不指定則訓練兩者。
    --features: 逗號分隔的特徵欄位名稱，不指定則使用全部預設特徵。
    """
    parser = argparse.ArgumentParser(description="Train CatBoost salary model")
    parser.add_argument("--target", choices=SALARY_BOUNDS, help="Train only one salary target")
    parser.add_argument("--features", help="Comma-separated X feature names from the frontend")
    args = parser.parse_args()

    run_training_pipeline(
        selected_targets=[args.target] if args.target else None,
        selected_features=parse_feature_list(args.features),
    )


if __name__ == "__main__":
    main()
