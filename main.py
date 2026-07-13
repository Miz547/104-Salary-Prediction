"""
全模型訓練主入口。

一次性訓練全部 6 個薪資預測模型：
- CatBoost × max 薪資 / min 薪資
- LightGBM × max 薪資 / min 薪資
- XGBoost  × max 薪資 / min 薪資

執行完畢後，每個模型都會產生：
1. PNG 視覺化圖表（薪資分布、預測比較、特徵重要性、Top tokens）
2. HTML 訓練報告
3. outputs/visualizations/training_summary.csv（所有模型的指標彙整）
4. data/clean data/ 的乾淨特徵 CSV（可供後續分析）

使用方式：
    在 Salary Predictive Analytics Engine/ 根目錄執行：
        python main.py

所屬位置：Salary Predictive Analytics Engine/main.py
"""

from pathlib import Path
import os

# 限制 joblib 平行處理的 CPU 核心數為 1，避免在 Windows 環境下產生子行程問題。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd

from config import CLEAN_DATA_DIR, DATA_FILE, LOG_FILE, SALARY_BOUNDS, VISUALIZATION_DIR
from models.CatBoost import train_catboost
from models.LightGBM import predict_lightgbm, train_lightgbm
from models.XGBoost import predict_xgboost, train_xgboost
from utils.cleaning import read_104_csv
from utils.engineering import build_model_frame, split_data
from utils.logger import setup_logger
from utils.visualization import evaluate_predictions, save_visualizations


# 三個模型的訓練與預測函式對應表。
# 新增模型時只需在此加入對應的 train / predict 函式即可。
MODEL_RUNNERS = {
    "CatBoost": {
        "train": train_catboost,
        # CatBoost 直接呼叫 model.predict，不需要額外的預處理。
        "predict": lambda model, x_test: model.predict(x_test),
    },
    "LightGBM": {
        "train": train_lightgbm,
        # LightGBM 需要對齊 Categorical categories，使用專用 predict 函式。
        "predict": predict_lightgbm,
    },
    "XGBoost": {
        "train": train_xgboost,
        # XGBoost 需要 OrdinalEncoder 轉換，使用專用 predict 函式。
        "predict": predict_xgboost,
    },
}


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
        f"{run_name} 訓練結果：\n"
        f"1. MAE={metrics['mae']:,.2f}, RMSE={metrics['rmse']:,.2f}, R2={metrics['r2']:.4f}\n"
        f"2. 最佳參數：{best_params}\n"
        f"3. 交叉驗證最佳 MAE(log)：{cv_mae_log:.4f}\n"
        f"4. MAPE：{metrics['mape']:.2f}%\n"
        f"5. 預測誤差 <= 25% 比例：{metrics['within_threshold_percent']:.2f}%"
    )


def save_clean_feature_data(df_model: pd.DataFrame, y: pd.Series, salary_bound: str, logger=None) -> Path:
    """
    將清洗後的特徵資料儲存為 CSV，供後續分析或除錯使用。

    輸出位置：data/clean data/104_salary_clean_features_{salary_bound}.csv
    CSV 包含所有衍生特徵欄位，以及 target_log1p 目標欄位。

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
    # 附加 log1p 目標欄位，方便後續分析。
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


def run_training_pipeline(data_file: Path = DATA_FILE) -> pd.DataFrame:
    """
    執行完整的全模型訓練流程。

    流程：
    1. 讀取原始 CSV 資料
    2. 對 max / min 兩個薪資目標分別：
       a. 建立特徵框架（清洗、聚類、衍生特徵）
       b. 儲存清洗後特徵 CSV
       c. 切分訓練/測試集
       d. 依序訓練 CatBoost / LightGBM / XGBoost
       e. 評估並產生視覺化報告
    3. 將所有模型的指標彙整為 training_summary.csv

    參數：
        data_file - 原始 CSV 資料路徑，預設使用 config.DATA_FILE。

    回傳：
        包含所有模型訓練結果的 DataFrame（同時儲存為 training_summary.csv）。
    """
    logger = setup_logger(log_file=LOG_FILE)
    logger.info("開始執行 104 薪資模型訓練流程")
    logger.info("設定：data_file=%s，visualization_dir=%s，log_file=%s", data_file, VISUALIZATION_DIR, LOG_FILE)

    df = read_104_csv(data_file, logger=logger)
    results = []

    for salary_bound in SALARY_BOUNDS:
        logger.info("========== 準備薪資%s值目標 ==========", salary_bound)
        x, y, feature_cols, token_counter, df_model = build_model_frame(df, salary_bound, logger=logger)
        save_clean_feature_data(df_model, y, salary_bound, logger=logger)
        x_train, x_test, y_train, y_test = split_data(x, y, logger=logger)

        for model_name, runner in MODEL_RUNNERS.items():
            run_name = f"{model_name}_{salary_bound}"
            logger.info("---------- 開始訓練 %s ----------", run_name)
            model, best_params, cv_mae_log = runner["train"](x_train, y_train, feature_cols, logger=logger)
            logger.info("模型評估：%s 使用測試集進行預測", run_name)
            y_pred_log = runner["predict"](model, x_test)
            metrics = evaluate_predictions(y_test, y_pred_log)

            logger.info("視覺化：%s 輸出薪資分布、預測比較、特徵重要性、Top tokens 與 HTML 報告", run_name)
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

            # 將此次訓練的完整結果存入清單，最後彙整為 DataFrame。
            result = {
                "target": salary_bound,
                "model": model_name,
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
            logger.info("報告輸出：%s", report_path.resolve())

    # 將所有模型的訓練指標儲存為 CSV 摘要檔。
    results_df = pd.DataFrame(results)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = VISUALIZATION_DIR / "training_summary.csv"
    results_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info("訓練摘要已輸出：%s", summary_path.resolve())
    logger.info("全部視覺化報告資料夾：%s", VISUALIZATION_DIR.resolve())
    logger.info("流程結束")
    return results_df


def main() -> None:
    """主函式：直接執行完整的全模型訓練流程。"""
    run_training_pipeline()


if __name__ == "__main__":
    main()
