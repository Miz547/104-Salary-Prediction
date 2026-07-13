"""
視覺化與模型評估工具模組。

提供訓練完成後的評估指標計算、PNG 圖表產生與 HTML 報告輸出功能。
所有圖表皆使用非互動式 Agg 後端（適合伺服器端無 GUI 環境）。

產出內容：
- 薪資分布直方圖（原始值 + log1p 轉換後）
- 預測值 vs 實際值散點圖
- 特徵重要性橫條圖
- Top 20 高頻 token 橫條圖
- 整合以上四張圖的 HTML 報告（含評估指標摘要）

所屬位置：Salary Predictive Analytics Engine/utils/visualization.py
被以下模組引用：main.py、catboost_main.py
"""

from html import escape
from pathlib import Path

import matplotlib

# 使用非互動式後端，避免在無 GUI 的伺服器環境中拋出錯誤。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import ERROR_THRESHOLD_PERCENT, TARGET_COLUMN

# 設定中文字型，優先使用微軟正黑體，備選 Arial Unicode MS。
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Arial Unicode MS", "sans-serif"]
# 避免負號顯示為方塊。
plt.rcParams["axes.unicode_minus"] = False


def evaluate_predictions(y_test_log, y_pred_log) -> dict:
    """
    計算模型在測試集上的各項評估指標。

    預測值與實際值皆以 log1p 形式傳入，此函式先以 expm1 還原為原始月薪，
    再計算各項指標（避免對 log 空間的誤差作解讀）。

    計算指標：
        mae                    - 平均絕對誤差（元）
        rmse                   - 均方根誤差（元）
        r2                     - 決定係數
        mape                   - 平均絕對百分比誤差（%）
        within_threshold_percent - 預測誤差在 ERROR_THRESHOLD_PERCENT% 以內的比例（%）
        y_true                 - 還原後的實際月薪陣列（供繪圖使用）
        y_pred                 - 還原後的預測月薪陣列（供繪圖使用）

    參數：
        y_test_log - 實際月薪的 log1p 值（Series 或 ndarray）。
        y_pred_log - 模型預測的 log1p 值（ndarray）。

    回傳：
        包含上述所有指標的 dict。
    """
    y_true = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)
    # 避免除以零：實際薪資為 0 的列以 NaN 替代，在 nanmean 中被略過。
    safe_y_true = np.where(y_true == 0, np.nan, y_true)
    absolute_percent_error = np.abs((y_true - y_pred) / safe_y_true) * 100
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.nanmean(absolute_percent_error)),
        "within_threshold_percent": float(
            np.nanmean(absolute_percent_error <= ERROR_THRESHOLD_PERCENT) * 100
        ),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _save_salary_distribution(df_model: pd.DataFrame, output_dir: Path, prefix: str) -> Path:
    """
    產生薪資分布直方圖並儲存為 PNG。

    左圖：原始月薪分布（通常呈右偏）。
    右圖：log1p 轉換後的分布（較接近常態）。
    兩圖並排，方便比較轉換前後的差異。

    參數：
        df_model   - 含 target_monthly_salary 欄位的訓練 DataFrame。
        output_dir - 輸出目錄 Path 物件。
        prefix     - 檔名前綴（如 'catboost_max'）。

    回傳：
        儲存完成的 PNG 檔案路徑。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    salary = df_model[TARGET_COLUMN]

    axes[0].hist(salary, bins=40, color="#4477AA", alpha=0.85)
    axes[0].set_title("Monthly Salary Distribution")
    axes[0].set_xlabel("Monthly salary")
    axes[0].set_ylabel("Count")

    axes[1].hist(np.log1p(salary), bins=40, color="#66AA55", alpha=0.85)
    axes[1].set_title("Log1p Monthly Salary")
    axes[1].set_xlabel("log1p(monthly salary)")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    path = output_dir / f"{prefix}_salary_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_prediction_plot(metrics: dict, output_dir: Path, prefix: str) -> Path:
    """
    產生「實際薪資 vs 預測薪資」散點圖並儲存為 PNG。

    若資料點超過 500 筆，只取前 500 筆繪圖（避免圖片過於密集）。
    虛線代表完美預測線（y=x），點越靠近虛線表示預測越準確。

    參數：
        metrics    - evaluate_predictions 回傳的 dict，需含 y_true 與 y_pred。
        output_dir - 輸出目錄 Path 物件。
        prefix     - 檔名前綴。

    回傳：
        儲存完成的 PNG 檔案路徑。
    """
    y_true = np.asarray(metrics["y_true"])
    y_pred = np.asarray(metrics["y_pred"])
    # 最多取前 500 個資料點，避免圖片過度密集影響可讀性。
    sample_size = min(500, len(y_true))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true[:sample_size], y_pred[:sample_size], alpha=0.55, color="#CC6677")
    lower = min(y_true.min(), y_pred.min())
    upper = max(y_true.max(), y_pred.max())
    # 虛線代表完美預測（預測值=實際值）。
    ax.plot([lower, upper], [lower, upper], color="#222222", linestyle="--", linewidth=1)
    ax.set_title("Actual vs Predicted Monthly Salary")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()

    path = output_dir / f"{prefix}_prediction_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_feature_importance(model, feature_cols: list[str], output_dir: Path, prefix: str) -> Path | None:
    """
    產生特徵重要性橫條圖並儲存為 PNG。

    優先使用 get_feature_importance()（CatBoost），
    其次使用 feature_importances_ 屬性（LightGBM / XGBoost / sklearn）。
    若模型不支援特徵重要性則跳過，回傳 None。

    只顯示前 20 名最重要的特徵，從下到上排列（最重要在最上方）。

    參數：
        model        - 訓練完成的模型物件。
        feature_cols - 特徵欄位名稱清單，順序須與模型訓練時一致。
        output_dir   - 輸出目錄 Path 物件。
        prefix       - 檔名前綴。

    回傳：
        儲存完成的 PNG 檔案路徑，或 None（模型不支援特徵重要性時）。
    """
    if hasattr(model, "get_feature_importance"):
        # CatBoost 專用介面。
        importances = model.get_feature_importance()
    elif hasattr(model, "feature_importances_"):
        # sklearn 通用介面（LightGBM / XGBoost 皆支援）。
        importances = model.feature_importances_
    else:
        return None

    importance_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    # 只取前 20 名，降序排列。
    importance_df = importance_df.sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    # 反轉順序讓最重要的特徵顯示在最上方。
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="#228833")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()

    path = output_dir / f"{prefix}_feature_importance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_top_tokens(token_counter, output_dir: Path, prefix: str) -> Path | None:
    """
    產生 Top 20 高頻 token 橫條圖並儲存為 PNG。

    token_counter 由 utils/engineering.py 的 build_tokenizer 建立，
    統計了全訓練集文字斷詞後的 token 出現次數。

    若 token_counter 為空（無文字特徵），則跳過，回傳 None。

    參數：
        token_counter - Counter 物件（token → 出現次數）。
        output_dir    - 輸出目錄 Path 物件。
        prefix        - 檔名前綴。

    回傳：
        儲存完成的 PNG 檔案路徑，或 None（無資料時）。
    """
    top_tokens = token_counter.most_common(20)
    if not top_tokens:
        return None

    token_df = pd.DataFrame(top_tokens, columns=["token", "count"])
    fig, ax = plt.subplots(figsize=(9, 6))
    # 反轉順序讓頻率最高的 token 顯示在最上方。
    ax.barh(token_df["token"][::-1], token_df["count"][::-1], color="#AA7744")
    ax.set_title("Top Text Tokens")
    ax.set_xlabel("Count")
    fig.tight_layout()

    path = output_dir / f"{prefix}_top_tokens.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_visualizations(
    model,
    df_model: pd.DataFrame,
    feature_cols: list[str],
    token_counter,
    metrics: dict,
    best_params: dict,
    output_dir: str | Path,
    prefix: str,
) -> Path:
    """
    產生全套視覺化圖表並輸出整合 HTML 報告。

    依序產生以下圖表（若不支援則略過）：
    1. 薪資分布圖
    2. 預測比較散點圖
    3. 特徵重要性圖
    4. Top tokens 圖

    最後將所有圖表嵌入 HTML 報告，並顯示評估指標卡片與最佳參數。

    參數：
        model        - 訓練完成的模型物件。
        df_model     - 含目標欄位的完整訓練 DataFrame。
        feature_cols - 特徵欄位名稱清單。
        token_counter - TF-IDF 斷詞後的 token 頻率統計 Counter。
        metrics      - evaluate_predictions 回傳的評估指標 dict。
        best_params  - 交叉驗證選出的最佳超參數 dict。
        output_dir   - 所有輸出檔案的目錄（不存在時自動建立）。
        prefix       - 所有輸出檔名的前綴（如 'catboost_max'）。

    回傳：
        HTML 報告檔案的 Path 物件。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 產生各類圖表，None（不支援）的項目在後續過濾掉。
    image_paths = [
        _save_salary_distribution(df_model, output_dir, prefix),
        _save_prediction_plot(metrics, output_dir, prefix),
        _save_feature_importance(model, feature_cols, output_dir, prefix),
        _save_top_tokens(token_counter, output_dir, prefix),
    ]
    # 過濾掉回傳 None 的項目（如模型不支援特徵重要性）。
    image_paths = [path for path in image_paths if path is not None]

    # 產生 HTML 報告，內嵌指標卡片、最佳參數與所有圖表。
    report_path = output_dir / f"{prefix}_report.html"
    image_html = "\n".join(
        f'<section><h2>{escape(path.stem)}</h2><img src="{escape(path.name)}" alt="{escape(path.stem)}"></section>'
        for path in image_paths
    )
    params_html = escape(str(best_params))
    report_path.write_text(
        f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="utf-8">
  <title>{escape(prefix)} salary model report</title>
  <style>
    body {{ font-family: "Microsoft JhengHei", Arial, sans-serif; margin: 32px; color: #222; }}
    h1 {{ margin-bottom: 8px; }}
    .metrics {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 24px 0; }}
    .metric {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; min-width: 150px; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 4px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
    section {{ margin: 28px 0; }}
    code {{ white-space: pre-wrap; }}
    .back-link {{ position: fixed; top: 16px; right: 16px; z-index: 20; padding: 10px 14px; border: 1px solid #fb923c; border-radius: 4px; background: #fff7ed; color: #c2410c; font-weight: 800; text-decoration: none; box-shadow: 0 2px 8px rgba(16,24,40,0.08); }}
    .back-link:hover {{ background: #f97316; border-color: #f97316; color: #fff; }}
  </style>
</head>
<body>
  <a class="back-link" href="../../salary_dashboard.html">薪資預測工作台</a>
  <h1>{escape(prefix)} Salary Model Report</h1>
  <div class="metrics">
    <div class="metric">MAE<strong>{metrics["mae"]:,.2f}</strong></div>
    <div class="metric">RMSE<strong>{metrics["rmse"]:,.2f}</strong></div>
    <div class="metric">R2<strong>{metrics["r2"]:.4f}</strong></div>
    <div class="metric">MAPE<strong>{metrics["mape"]:.2f}%</strong></div>
    <div class="metric">Error <= {ERROR_THRESHOLD_PERCENT}%<strong>{metrics["within_threshold_percent"]:.2f}%</strong></div>
  </div>
  <h2>Best Parameters</h2>
  <code>{params_html}</code>
  {image_html}
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path
