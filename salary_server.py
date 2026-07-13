"""
104 Salary On-Prem Server
合併版：整合 Salary Predictive Analytics Engine 的 CatBoost / LightGBM / XGBoost 三模型
預測薪資範圍（最低 ~ 最高）
"""
import os
import re
import sys
import csv
import json
import posixpath
import secrets
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPConnection
from http.cookies import SimpleCookie
from html import unescape
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import Request, urlopen

# Ensure the script's own directory is on sys.path so subpackages (utils, models, config) are importable
# regardless of the working directory when launched from ServerManager or start_all.bat
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from config import (
    RANDOM_STATE, TEXT_CLUSTER_COUNT, TFIDF_MAX_FEATURES, TARGET_COLUMN,
)
from models.CatBoost import train_catboost
from models.LightGBM import train_lightgbm, predict_lightgbm
from models.XGBoost import train_xgboost, predict_xgboost
from utils.cleaning import (
    read_104_csv, fill_required_columns, filter_tech_jobs, add_salary_target,
)
from utils.engineering import normalize_experience_band, build_tokenizer, fallback_chinese_tokenizer

# ─── 路徑設定 ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
STATIC_DIR  = BASE_DIR / "public"
DATA_PATH   = BASE_DIR / "data" / "104_0605.csv"
MODEL_DIR   = BASE_DIR / "outputs" / "trained_models"
PRIVATE_DIR = BASE_DIR / "_private"
DASHBOARD_UPSTREAM   = ("127.0.0.1", 8011)
RAG_UPSTREAM         = ("127.0.0.1", 8050)

HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}

SALARY_BOUNDS    = ["max", "min"]
MODEL_NAMES      = ["catboost", "lightgbm", "xgboost"]
EXPERIENCE_BANDS = ["經歷不拘~1年", "1~3年", "3~5年", "5~7年", "7年以上"]

FEATURE_CANDIDATES = [
    "keyword", "clean_location", "jobCompanyName", "jobCompanyIndustry",
    "employee_scale_cat", "experience_band", "jobRqDepartment",
    "job_cluster_label", "experience_band_cluster", "is_tech_job", "announce_date_cat",
]

# 全域快取：首次載入後不再重訓
_model_bundles: dict = {}

# ─── 變數模擬器：CatBoost 訓練狀態 ──────────────────────────────────────────
CATBOOST_MAIN  = BASE_DIR / "catboost_main.py"
SUMMARY_PATH   = BASE_DIR / "outputs" / "visualizations" / "training_summary.csv"
_train_lock    = threading.Lock()
_train_state: dict = {
    "running":     False,
    "started_at":  None,
    "finished_at": None,
    "return_code": None,
    "message":     "Ready",
    "stdout":      "",
    "stderr":      "",
    "payload":     {},
}


def _read_training_summary() -> list[dict]:
    if not SUMMARY_PATH.exists():
        return []
    with SUMMARY_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _run_catboost_job(payload: dict) -> None:
    target   = str(payload.get("target") or "").strip()
    features = payload.get("x_features") or payload.get("features") or []
    features = [str(f).strip() for f in features if str(f).strip()]
    command  = [sys.executable, str(CATBOOST_MAIN)]
    if target:
        command.extend(["--target", target])
    if features:
        command.extend(["--features", ",".join(features)])

    with _train_lock:
        _train_state.update({
            "running": True, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": None, "return_code": None,
            "message": "Training CatBoost...", "stdout": "", "stderr": "",
            "payload": {"target": target, "x_features": features,
                        "command": " ".join(command)},
        })

    proc = subprocess.run(
        command, cwd=str(BASE_DIR),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )

    with _train_lock:
        _train_state.update({
            "running": False, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "return_code": proc.returncode,
            "message": "Training completed" if proc.returncode == 0 else "Training failed",
            "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:],
        })


def get_train_status() -> dict:
    with _train_lock:
        state = dict(_train_state)
    return {"status": "running" if state["running"] else "idle",
            **state, "summary": _read_training_summary()}


def start_training(payload: dict) -> dict:
    with _train_lock:
        if _train_state["running"]:
            return {"status": "running", **_train_state,
                    "summary": _read_training_summary()}
    t = threading.Thread(target=_run_catboost_job, args=(payload,), daemon=True)
    t.start()
    time.sleep(0.1)
    return {"status": "started", **_train_state,
            "summary": _read_training_summary()}


# ─── 工具函式 ─────────────────────────────────────────────────────────────────

def normalized_url_path(path):
    raw = urlsplit(path).path
    decoded = unquote(raw).replace("\\", "/")
    normalized = posixpath.normpath(decoded.lstrip("/"))
    return "/" if normalized == "." else "/" + normalized


def read_text_secret(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()



def split_category_parts(category):
    return [part.strip() for part in str(category).split(",") if part.strip()]


def is_displayable_category(category):
    category = str(category).strip()
    if not category or len(category) > 42:
        return False
    return bool(re.search(r"[一-鿿A-Za-z]", category))


def search_tokens(text):
    text = str(text).lower().strip()
    ascii_tokens = re.findall(r"[a-z0-9_#+.]+", text)
    cjk_chars    = re.findall(r"[一-鿿]", text)
    cjk_bigrams  = [a + b for a, b in zip(cjk_chars, cjk_chars[1:])]
    return set(ascii_tokens + cjk_bigrams + cjk_chars)


def score_related_category(title, category, source_title="", count=0):
    t  = str(title).lower().strip()
    c  = str(category).lower().strip()
    st = str(source_title).lower().strip()
    tt = search_tokens(t)
    ct = search_tokens(c)
    ss = search_tokens(st)
    score = 0.0
    if t and t in c:  score += 80
    if c and c in t:  score += 120
    if t and st and t in st: score += 80
    score += len(tt & ct) * 18
    score += len(tt & ss) * 10
    score += min(float(count), 80.0) * 0.2
    return score


# ─── 訓練流程 ─────────────────────────────────────────────────────────────────

def _fit_text_clusters(df):
    """對 df 進行 TF-IDF + KMeans 分群，回傳 (df_with_cluster, vectorizer, kmeans)。"""
    df = df.copy()
    df["clean_category"]    = df["jobCategory"].astype(str).str.replace(r"[,，/|;]+", " ", regex=True)
    df["job_text_features"] = df["jobTitles"].astype(str) + " " + df["clean_category"]

    token_counter = Counter()
    vectorizer = TfidfVectorizer(
        tokenizer=build_tokenizer(token_counter),
        max_features=TFIDF_MAX_FEATURES,
        lowercase=False,
        token_pattern=None,
    )
    x_tfidf = vectorizer.fit_transform(df["job_text_features"])
    cluster_count = min(TEXT_CLUSTER_COUNT, len(df))
    kmeans = KMeans(n_clusters=cluster_count, random_state=RANDOM_STATE, n_init=10)
    df["job_cluster_label"] = kmeans.fit_predict(x_tfidf).astype(str)
    # 將 closure tokenizer 換成模組層級函式，使 vectorizer 可被 pickle 儲存
    vectorizer.tokenizer = fallback_chinese_tokenizer
    return df, vectorizer, kmeans


def _add_derived_features(df_model):
    df = df_model.copy()
    df["clean_location"]        = df["jobLocationAt"].astype(str).str.strip().str[:3]
    df["employee_scale_cat"]    = df["employeeCount"].astype(str)
    df["announce_date_cat"]     = df["jobAnnounceDate"].astype(str)
    df["experience_band"]       = df["jobRqYear"].apply(normalize_experience_band)
    df["experience_band_cluster"] = df["experience_band"] + "_" + df["job_cluster_label"]
    return df


def _build_xy(df_model):
    feature_cols = [c for c in FEATURE_CANDIDATES if c in df_model.columns]
    x = df_model[feature_cols].copy().astype(str)
    y = np.log1p(df_model[TARGET_COLUMN])
    return x, y, feature_cols


def _dashboard_artifacts(df_model, cb_model, feature_cols):
    cluster_stats = (
        df_model.groupby("job_cluster_label")[TARGET_COLUMN]
        .mean().sort_index().round().astype(int).to_dict()
    )
    cluster_stats = {f"職務群組 {k}": v for k, v in cluster_stats.items()}

    loc_stats = (
        df_model.groupby("clean_location")[TARGET_COLUMN]
        .mean().sort_values(ascending=False).head(12).round().astype(int).to_dict()
    )

    cat_frame = (
        df_model.groupby("jobCategory")[TARGET_COLUMN]
        .agg(["min", "mean", "max", "count"])
        .query("count >= 3")
        .sort_values("mean", ascending=False)
        .head(80).round().astype(int)
    )
    category_stats = {
        name: {"min": int(r["min"]), "mean": int(r["mean"]),
               "max": int(r["max"]), "count": int(r["count"])}
        for name, r in cat_frame.iterrows()
    }

    importances    = cb_model.get_feature_importance()
    feature_weights = {
        feat: round(float(w), 2)
        for feat, w in zip(feature_cols, importances)
    }
    return cluster_stats, loc_stats, category_stats, feature_weights


def _category_search_artifacts(df_model):
    counts, pairs = {}, []
    for _, row in df_model[["jobTitles", "jobCategory"]].dropna().iterrows():
        title    = str(row["jobTitles"])
        category = str(row["jobCategory"])
        pairs.append((title, category))
        for part in split_category_parts(category):
            if is_displayable_category(part):
                counts[part] = counts.get(part, 0) + 1

    popular = [p for p, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)][:300]
    return {"category_part_counts": counts,
            "popular_category_parts": popular,
            "title_category_pairs": pairs}


def _prepare_for_bound(salary_bound):
    """讀取資料、清洗、分群、建立特徵，回傳訓練所需全部物件。"""
    df = read_104_csv(DATA_PATH)
    df = fill_required_columns(df)
    df = filter_tech_jobs(df)
    df, vectorizer, kmeans = _fit_text_clusters(df)
    df_model = add_salary_target(df, salary_bound)
    df_model = _add_derived_features(df_model)
    x, y, feature_cols = _build_xy(df_model)
    return df_model, x, y, feature_cols, vectorizer, kmeans


def train_and_save_all():
    """訓練全部 6 個模型（3種 × max/min），儲存 artifacts。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60, flush=True)
    print("開始訓練全部模型（首次啟動或模型不存在時執行）", flush=True)

    for salary_bound in SALARY_BOUNDS:
        print(f"\n[{salary_bound.upper()}] 準備資料...", flush=True)
        df_model, x, y, feature_cols, vectorizer, kmeans = _prepare_for_bound(salary_bound)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=RANDOM_STATE
        )

        print(f"[{salary_bound.upper()}] 訓練 CatBoost...", flush=True)
        cb_model, _, _ = train_catboost(x_train, y_train, feature_cols)

        print(f"[{salary_bound.upper()}] 訓練 LightGBM...", flush=True)
        lgb_model, _, _ = train_lightgbm(x_train, y_train, feature_cols)

        print(f"[{salary_bound.upper()}] 訓練 XGBoost...", flush=True)
        xgb_model, _, _ = train_xgboost(x_train, y_train, feature_cols)

        actual = np.expm1(y_test.to_numpy())
        maes = {
            "catboost": float(np.mean(np.abs(actual - np.expm1(cb_model.predict(x_test))))),
            "lightgbm": float(np.mean(np.abs(actual - np.expm1(predict_lightgbm(lgb_model, x_test))))),
            "xgboost":  float(np.mean(np.abs(actual - np.expm1(predict_xgboost(xgb_model, x_test))))),
        }
        print(f"[{salary_bound.upper()}] MAE → {maes}", flush=True)

        cluster_stats, loc_stats, category_stats, feature_weights = \
            _dashboard_artifacts(df_model, cb_model, feature_cols)

        artifacts = {
            "vectorizer":      vectorizer,
            "kmeans":          kmeans,
            "feature_cols":    feature_cols,
            "maes":            maes,
            "trained_rows":    len(df_model),
            "catboost":        cb_model,
            "lightgbm":        lgb_model,
            "xgboost":         xgb_model,
            "cluster_stats":   cluster_stats,
            "loc_stats":       loc_stats,
            "category_stats":  category_stats,
            "feature_weights": feature_weights,
            "locations":       sorted(df_model["clean_location"].dropna().astype(str).unique().tolist()),
            "experience_bands": EXPERIENCE_BANDS,
            "categories":      sorted(df_model["jobCategory"].dropna().astype(str).unique().tolist())[:1000],
        }
        artifacts.update(_category_search_artifacts(df_model))
        joblib.dump(artifacts, MODEL_DIR / f"artifacts_{salary_bound}.joblib")
        print(f"[{salary_bound.upper()}] 已儲存 artifacts_{salary_bound}.joblib", flush=True)

    print("\n全部模型訓練完成！", flush=True)


def load_model_bundles():
    """載入或訓練所有模型，回傳全域快取字典。"""
    global _model_bundles
    if _model_bundles:
        return _model_bundles

    all_exist = all(
        (MODEL_DIR / f"artifacts_{b}.joblib").exists()
        for b in SALARY_BOUNDS
    )
    if not all_exist:
        train_and_save_all()

    for b in SALARY_BOUNDS:
        _model_bundles[b] = joblib.load(MODEL_DIR / f"artifacts_{b}.joblib")

    return _model_bundles


# ─── 推論 ─────────────────────────────────────────────────────────────────────

def _build_inference_row(payload, bundle):
    """
    index.html 格式 → 從原始欄位建構特徵列
    payload 欄位：jobLocationAt, jobTitles, jobCategory, jobRqYear,
                  keyword, jobCompanyName, jobCompanyIndustry, jobRqDepartment, employeeCount
    """
    location   = str(payload.get("jobLocationAt", "")).strip()[:3] or "Unknown"
    title      = str(payload.get("jobTitles", "")).strip()          or "Unknown"
    category   = str(payload.get("jobCategory", "")).strip()        or "Unknown"
    rq_year    = str(payload.get("jobRqYear", "")).strip()          or "Unknown"
    keyword    = str(payload.get("keyword", "")).strip()            or "Unknown"
    company    = str(payload.get("jobCompanyName", "")).strip()     or "Unknown"
    industry   = str(payload.get("jobCompanyIndustry", "")).strip() or "Unknown"
    department = str(payload.get("jobRqDepartment", "")).strip()    or "Unknown"
    emp_count  = str(payload.get("employeeCount", "0")).strip()     or "0"

    clean_cat = re.sub(r"[,，/|;]+", " ", category)
    job_text  = f"{title} {clean_cat}"
    cluster   = str(bundle["kmeans"].predict(bundle["vectorizer"].transform([job_text]))[0])
    exp_band  = normalize_experience_band(rq_year)

    row_data = {
        "keyword":                 keyword,
        "clean_location":          location,
        "jobCompanyName":          company,
        "jobCompanyIndustry":      industry,
        "employee_scale_cat":      emp_count,
        "experience_band":         exp_band,
        "jobRqDepartment":         department,
        "job_cluster_label":       cluster,
        "experience_band_cluster": f"{exp_band}_{cluster}",
        "is_tech_job":             "1",
        "announce_date_cat":       "Unknown",
    }
    feature_cols = bundle["feature_cols"]
    df_row = pd.DataFrame(
        [{col: row_data.get(col, "Unknown") for col in feature_cols}],
        columns=feature_cols,
    ).astype(str)
    return df_row, cluster, exp_band


def _build_dashboard_row(payload, bundle):
    """
    salary_dashboard.html 格式 → features 已是處理後欄位
    payload 欄位：model, target, job_title,
                  features.{keyword, clean_location, experience_band,
                             jobRqDepartment, jobCompanyIndustry, employee_scale_cat}
    """
    feats      = payload.get("features", {})
    title      = str(payload.get("job_title", "")).strip() or "Unknown"
    keyword    = str(feats.get("keyword", "")).strip()           or "Unknown"
    location   = str(feats.get("clean_location", "")).strip()[:3] or "Unknown"
    exp_band   = str(feats.get("experience_band", "")).strip()   or "經歷不拘~1年"
    department = str(feats.get("jobRqDepartment", "")).strip()   or "Unknown"
    industry   = str(feats.get("jobCompanyIndustry", "")).strip() or "Unknown"
    emp_count  = str(feats.get("employee_scale_cat", "0")).strip() or "0"

    # 用 job_title + department 做文字分群
    job_text = f"{title} {department}"
    cluster  = str(bundle["kmeans"].predict(bundle["vectorizer"].transform([job_text]))[0])

    row_data = {
        "keyword":                 keyword,
        "clean_location":          location,
        "jobCompanyName":          "Unknown",
        "jobCompanyIndustry":      industry,
        "employee_scale_cat":      emp_count,
        "experience_band":         exp_band,
        "jobRqDepartment":         department,
        "job_cluster_label":       cluster,
        "experience_band_cluster": f"{exp_band}_{cluster}",
        "is_tech_job":             "1",
        "announce_date_cat":       "Unknown",
    }
    feature_cols = bundle["feature_cols"]
    df_row = pd.DataFrame(
        [{col: row_data.get(col, "Unknown") for col in feature_cols}],
        columns=feature_cols,
    ).astype(str)
    return df_row


_MODEL_KEY_MAP = {"CatBoost": "catboost", "LightGBM": "lightgbm", "XGBoost": "xgboost"}


def get_prediction_payload(payload):
    """
    自動判斷 payload 格式：
    - dashboard 格式（含 model / target / features 欄位）→ 指定模型+目標，回傳單一薪資
    - index 格式（含 jobTitles / jobLocationAt 等）→ CatBoost min/max 範圍
    """
    bundles = load_model_bundles()

    # ── dashboard 格式 ──────────────────────────────────────────────────────
    if "model" in payload and "target" in payload and "features" in payload:
        model_key = _MODEL_KEY_MAP.get(payload["model"], "catboost")
        bound     = payload["target"] if payload["target"] in SALARY_BOUNDS else "max"
        bundle    = bundles[bound]
        row       = _build_dashboard_row(payload, bundle)

        if model_key == "catboost":
            salary = float(np.expm1(bundle["catboost"].predict(row)[0]))
        elif model_key == "lightgbm":
            salary = float(np.expm1(predict_lightgbm(bundle["lightgbm"], row)[0]))
        else:
            salary = float(np.expm1(predict_xgboost(bundle["xgboost"], row)[0]))

        mae = bundles[bound]["maes"].get(model_key, 0)
        return {
            "estimated_monthly_salary": round(salary),
            "reference_mae":            round(mae, 2),
            "source":                   "api",
        }

    # ── index.html 格式：CatBoost min + max 範圍 ───────────────────────────
    cluster_label = exp_band_label = None
    cb_min = cb_max = 0.0

    for bound in SALARY_BOUNDS:
        bundle = bundles[bound]
        row, cluster_label, exp_band_label = _build_inference_row(payload, bundle)
        val = float(np.expm1(bundle["catboost"].predict(row)[0]))
        if bound == "min":
            cb_min = val
        else:
            cb_max = val

    min_sal = round(cb_min)
    max_sal = round(cb_max)
    return {
        "predicted_min_salary":  min_sal,
        "predicted_max_salary":  max_sal,
        "predicted_salary_text": f"NT$ {min_sal:,} ~ NT$ {max_sal:,}",
        "features": {
            "clean_location":    str(payload.get("jobLocationAt", ""))[:3] or "Unknown",
            "job_cluster_label": cluster_label,
            "experience_band":   exp_band_label,
        },
        "input": payload,
    }


def get_health_payload():
    bundles = load_model_bundles()
    b = bundles["max"]
    mae_map = {
        f"{model}_{bound}": round(bundles[bound]["maes"][model], 2)
        for bound in SALARY_BOUNDS for model in MODEL_NAMES
    }
    return {"status": "ok", "trained_rows": b["trained_rows"], "mae": mae_map}


def get_options_payload():
    bundles = load_model_bundles()
    b = bundles["max"]
    return {
        "locations":       b["locations"],
        "experience_bands": b["experience_bands"],
        "categories":      b["categories"],
    }


def get_stats_payload():
    bundles = load_model_bundles()
    b = bundles["max"]
    return {
        "cluster_stats":   b["cluster_stats"],
        "loc_stats":       b["loc_stats"],
        "category_stats":  b["category_stats"],
        "feature_weights": b["feature_weights"],
    }


def get_related_categories_payload(title, limit=12):
    bundles = load_model_bundles()
    bundle = bundles["max"]
    scores = {}

    for src_title, full_cat in bundle.get("title_category_pairs", []):
        base = score_related_category(title, full_cat, src_title)
        if base <= 0:
            continue
        for part in split_category_parts(full_cat):
            if not is_displayable_category(part):
                continue
            cnt   = bundle.get("category_part_counts", {}).get(part, 0)
            pscore = score_related_category(title, part, src_title, cnt)
            scores[part] = max(scores.get(part, 0), base + pscore)

    if not scores:
        for part in bundle.get("popular_category_parts", []):
            cnt = bundle.get("category_part_counts", {}).get(part, 0)
            scores[part] = score_related_category(title, part, "", cnt)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        ranked = [(p, 0) for p in bundle.get("popular_category_parts", [])]

    return {"title": title, "categories": [c for c, _ in ranked[:limit]]}


# ─── 104 職缺抓取 ─────────────────────────────────────────────────────────────

def _request_text(url, headers=None, timeout=20):
    req_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/149 Safari/537.36",
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
    }
    if headers:
        req_headers.update(headers)
    req = Request(url, headers=req_headers)
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="ignore")


def _extract_104_job_id(url):
    m = re.search(r"104\.com\.tw/job/([A-Za-z0-9]+)", url)
    return m.group(1) if m else ""


def _clean_html_text(value):
    value = re.sub(r"<[^>]+>", " ", str(value))
    value = unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def _normalize_104_payload(raw):
    data      = raw.get("data", raw)
    header    = data.get("header", {})    if isinstance(data, dict) else {}
    detail    = data.get("jobDetail", {}) if isinstance(data, dict) else {}
    condition = data.get("condition", {}) if isinstance(data, dict) else {}
    industry  = detail.get("custIndustry") or header.get("custIndustry") or data.get("industry") or ""
    employee  = (
        header.get("empNo") or header.get("custEmpNo") or
        data.get("employeeCount") or data.get("empNo") or "50"
    )
    location  = detail.get("addressRegion") or detail.get("addressArea") or detail.get("address") or ""
    cats      = detail.get("jobCategory") or detail.get("jobCategoryDesc") or []
    if isinstance(cats, list):
        category_text = ",".join(
            item.get("description", item.get("desc", "")) if isinstance(item, dict) else str(item)
            for item in cats
        )
    else:
        category_text = str(cats)

    rq_year = condition.get("workExp") or condition.get("workExpDesc") or detail.get("workExp") or "經歷不拘~1年"
    title   = header.get("jobName") or data.get("jobName") or raw.get("title") or ""
    company = header.get("custName") or data.get("custName") or raw.get("company") or ""
    content = _clean_html_text(detail.get("jobDescription") or data.get("jobDescription") or raw.get("description") or "")
    return {
        "jobLocationAt":      _clean_html_text(location)[:3] or "Unknown",
        "jobTitles":          _clean_html_text(title),
        "jobCategory":        _clean_html_text(category_text) or _clean_html_text(industry) or "Unknown",
        "jobRqYear":          _clean_html_text(rq_year) or "經歷不拘~1年",
        "jobCompanyName":     _clean_html_text(company),
        "jobCompanyIndustry": _clean_html_text(industry),
        "jobContent":         content,
        "employeeCount":      _clean_html_text(str(employee)),
        "jobRqDepartment":    _clean_html_text(condition.get("major", "") or condition.get("edu", "") or "Unknown"),
    }


def _parse_html_job_payload(html_text):
    title = company = description = ""
    for block in re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html_text, flags=re.I | re.S,
    ):
        try:
            parsed = json.loads(unescape(block.strip()))
        except Exception:
            continue
        for item in (parsed if isinstance(parsed, list) else [parsed]):
            if not isinstance(item, dict):
                continue
            if item.get("@type") in ("JobPosting", "Organization") or item.get("title"):
                title       = title       or _clean_html_text(item.get("title", ""))
                description = description or _clean_html_text(item.get("description", ""))
                hiring = item.get("hiringOrganization", {})
                if isinstance(hiring, dict):
                    company = company or _clean_html_text(hiring.get("name", ""))
    if not title:
        m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', html_text, re.I)
        if m:
            title = _clean_html_text(m.group(1))
    return {
        "jobLocationAt": "Unknown", "jobTitles": title, "jobCategory": "Unknown",
        "jobRqYear": "經歷不拘~1年", "jobCompanyName": company, "jobCompanyIndustry": "",
        "jobContent": description, "employeeCount": "50", "jobRqDepartment": "Unknown",
    }


def _fetch_104_job_payload(url):
    job_id = _extract_104_job_id(url)
    errors = []
    if job_id:
        try:
            raw = _request_text(
                f"https://www.104.com.tw/job/ajax/content/{job_id}",
                headers={"Accept": "application/json, text/plain, */*", "Referer": url},
            )
            return _normalize_104_payload(json.loads(raw)), "104 ajax content", errors
        except Exception as exc:
            errors.append(f"104 ajax failed: {exc}")
    try:
        html_text = _request_text(url)
        return _parse_html_job_payload(html_text), "104 html fallback", errors
    except Exception as exc:
        errors.append(f"html fetch failed: {exc}")
        raise RuntimeError(f"無法取得職缺資料：{'; '.join(errors)}")


def get_url_prediction_payload(payload):
    url = str(payload.get("url", "")).strip()
    if not url:
        raise ValueError("請輸入 104 職缺網址")
    if "104.com.tw/job/" not in url:
        raise ValueError("請輸入有效的 104 職缺網址")

    target_mode = str(payload.get("target", "range")).strip().lower()
    if target_mode not in {"range", "min", "max"}:
        raise ValueError("target must be range, min, or max")
    model_name = str(payload.get("model", "CatBoost")).strip()
    if model_name not in {"CatBoost", "LightGBM", "XGBoost"}:
        raise ValueError("model must be CatBoost, LightGBM, or XGBoost")

    job, source, warnings = _fetch_104_job_payload(url)

    def _make_pred_payload(bound):
        return {
            "model": model_name,
            "target": bound,
            "job_title": job.get("jobTitles", ""),
            "features": {
                "keyword":              job.get("jobCategory", "") or job.get("jobTitles", ""),
                "clean_location":       (job.get("jobLocationAt", "") or "")[:3],
                "experience_band":      normalize_experience_band(job.get("jobRqYear", "")),
                "jobRqDepartment":      job.get("jobCategory", ""),
                "jobCompanyIndustry":   job.get("jobCompanyIndustry", ""),
                "employee_scale_cat":   job.get("employeeCount", "50"),
            },
        }

    item = {"model": model_name, "target": target_mode, "payloads": {}}
    min_salary = max_salary = 0

    if target_mode in {"range", "min"}:
        min_p = _make_pred_payload("min")
        min_result = get_prediction_payload(min_p)
        item["min_prediction"] = min_result
        item["payloads"]["min"] = min_p
        min_salary = min_result.get("estimated_monthly_salary", 0)

    if target_mode in {"range", "max"}:
        max_p = _make_pred_payload("max")
        max_result = get_prediction_payload(max_p)
        item["max_prediction"] = max_result
        item["payloads"]["max"] = max_p
        max_salary = max_result.get("estimated_monthly_salary", 0)

    if target_mode == "range":
        low, high = min(min_salary, max_salary), max(min_salary, max_salary)
        item["range"] = {"low": low, "high": high, "text": f"NT$ {low:,} - {high:,}"}
    else:
        salary = min_salary if target_mode == "min" else max_salary
        item["range"] = {"low": salary, "high": salary, "text": f"NT$ {salary:,}"}

    return {
        "url": url,
        "target": target_mode,
        "source": source,
        "warnings": warnings,
        "job": job,
        "model_info": {
            "source": "Salary Predictive Analytics Engine / salary_server.py",
            "note": "整合 CatBoost / LightGBM / XGBoost 三模型，直接呼叫本機模型。",
            "models": ["CatBoost", "LightGBM", "XGBoost"],
            "targets": ["min", "max"],
        },
        "predictions": [item],
    }


# ─── HTTP Handler ─────────────────────────────────────────────────────────────

class SalaryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def send_proxy_error(self, service_name, error):
        return self.send_json(
            {"error": f"{service_name} service unavailable",
             "detail": str(error),
             "hint": "請確認對應的本機服務已啟動。"},
            status=502,
        )

    def proxy_request(self, upstream, prefix, service_name):
        parsed = urlsplit(self.path)
        upstream_path = parsed.path[len(prefix):] or "/"
        if not upstream_path.startswith("/"):
            upstream_path = "/" + upstream_path
        if parsed.query:
            upstream_path += "?" + parsed.query

        body = None
        if self.command in ("POST", "PUT", "PATCH"):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else None

        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS and k.lower() != "host"
        }
        headers["Host"] = f"{upstream[0]}:{upstream[1]}"

        conn = None
        try:
            conn = HTTPConnection(upstream[0], upstream[1], timeout=120)
            conn.request(self.command, upstream_path, body=body, headers=headers)
            resp = conn.getresponse()
            body_bytes = resp.read()
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() in HOP_BY_HOP_HEADERS or k.lower() == "content-length":
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)
        except OSError as e:
            return self.send_proxy_error(service_name, e)
        finally:
            if conn:
                try: conn.close()
                except Exception: pass

    def do_GET(self):
        route    = normalized_url_path(self.path)
        raw_path = urlsplit(self.path).path.replace("\\", "/")

        if route == "/_private" or route.startswith("/_private/"):
            return self.send_error(404)
        if route == "/dashboard" and not raw_path.endswith("/"):
            return self.redirect("/dashboard/")
        if route in ("/admin.html", "/rag/admin.html"):
            return self.redirect("/rag/")
        if route == "/rag" and not raw_path.endswith("/"):
            return self.redirect("/rag/")
        if route.startswith("/dashboard/") or raw_path.startswith("/dashboard/"):
            return self.proxy_request(DASHBOARD_UPSTREAM, "/dashboard", "dashboard")
        if route.startswith("/rag/") or raw_path.startswith("/rag/"):
            return self.proxy_request(RAG_UPSTREAM, "/rag", "RAG")
        if route in ("/salary-url", "/salary-url/"):
            return self.redirect("/salary-url.html")
        if route in ("/", "/index.html", "/Index.html"):
            self.path = "/index.html"
            return super().do_GET()
        if route == "/health":
            return self.send_json(get_health_payload())
        if route == "/train-status":
            return self.send_json(get_train_status())
        if route == "/options":
            return self.send_json(get_options_payload())
        if route == "/stats":
            return self.send_json(get_stats_payload())
        if route == "/related-categories":
            query = parse_qs(urlsplit(self.path).query)
            title = query.get("title", [""])[0]
            limit = int(query.get("limit", ["12"])[0])
            return self.send_json(get_related_categories_payload(title, limit))
        return super().do_GET()

    def do_POST(self):
        route = normalized_url_path(self.path)
        if route.startswith("/dashboard/"):
            return self.proxy_request(DASHBOARD_UPSTREAM, "/dashboard", "dashboard")
        if route.startswith("/rag/"):
            return self.proxy_request(RAG_UPSTREAM, "/rag", "RAG")
        if route == "/run-main":
            length   = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw_body.decode("utf-8"))
                return self.send_json(start_training(payload))
            except Exception as exc:
                return self.send_json({"error": str(exc)}, status=500)
        if route not in ("/predict", "/predict-url"):
            return self.send_json({"error": "not found"}, status=404)

        length   = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
            if route == "/predict-url":
                return self.send_json(get_url_prediction_payload(payload))
            return self.send_json(get_prediction_payload(payload))
        except Exception as exc:
            return self.send_json({"error": str(exc)}, status=500)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - [{self.log_date_time_string()}] {fmt % args}", flush=True)


# ─── 進入點 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    server = ThreadingHTTPServer(("127.0.0.1", port), SalaryHandler)
    print(f"104 Salary AI Server（三模型整合版）running at http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
