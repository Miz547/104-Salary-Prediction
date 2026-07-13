"""
特徵工程工具模組。

將清洗後的原始 DataFrame 轉換成可直接送入機器學習模型的特徵矩陣。

主要流程：
1. normalize_experience_band  - 將年資文字標準化為前端對齊的區間標籤
2. build_tokenizer / fallback_chinese_tokenizer - 中文斷詞（優先 jieba，備選 bigram）
3. add_text_clusters          - TF-IDF + KMeans 對職缺文字進行聚類
4. build_model_frame          - 整合清洗、聚類、目標欄位，產出訓練用 X / y
5. split_data                 - 依設定比例切分訓練集與測試集
6. ordinal_encode_data        - OrdinalEncoder 對類別欄位編碼（供 XGBoost 使用）
7. align_categorical_columns  - 對齊 train/val/test 的 Categorical 類別（供 LightGBM 使用）

所屬位置：Salary Predictive Analytics Engine/utils/engineering.py
被以下模組引用：main.py、catboost_main.py
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE, TEXT_CLUSTER_COUNT, TFIDF_MAX_FEATURES
from utils.cleaning import add_salary_target, fill_required_columns, filter_tech_jobs

try:
    import jieba
except ImportError:  # pragma: no cover - optional dependency
    # jieba 未安裝時，改用 fallback_chinese_tokenizer（bigram 方法）。
    jieba = None


def normalize_experience_band(value) -> str:
    """
    將原始 jobRqYear 欄位值標準化為前端對齊的年資區間字串。

    原始資料的年資格式多樣（如「1年以上」「3~5年」「經歷不拘」），
    此函式統一映射為五個固定區間，與前端下拉選單選項保持一致。

    對應規則：
        - 空值、nan、None → '經歷不拘~1年'
        - 包含「不拘」「無經驗」「未滿1」等關鍵字 → '經歷不拘~1年'
        - 最小年資數字 < 1 → '經歷不拘~1年'
        - 最小年資 1~2 → '1~3年'
        - 最小年資 3~4 → '3~5年'
        - 最小年資 5~6 → '5~7年'
        - 最小年資 >= 7 → '7年以上'

    參數：
        value - jobRqYear 欄位的原始值（任意型別，會先轉為字串）。

    回傳：
        標準化後的年資區間字串。
    """
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return "經歷不拘~1年"

    low_experience_keywords = ["經歷不拘", "無經驗", "不拘", "未滿1", "1年以下"]
    if any(keyword in text for keyword in low_experience_keywords):
        return "經歷不拘~1年"

    # 從文字中提取所有數字，取最小值代表最低年資要求。
    numbers = [int(number) for number in re.findall(r"\d+", text)]
    if not numbers:
        return "經歷不拘~1年"

    years = min(numbers)
    if years < 1:
        return "經歷不拘~1年"
    if years < 3:
        return "1~3年"
    if years < 5:
        return "3~5年"
    if years < 7:
        return "5~7年"
    return "7年以上"


def fallback_chinese_tokenizer(text: str) -> list[str]:
    """
    不依賴 jieba 的備用中文斷詞函式。

    當 jieba 未安裝時使用此函式。
    策略：擷取英數字詞 + 中文 bigram（相鄰兩字組合），
    過濾掉長度 <= 1 的 token，降低雜訊。

    參數：
        text - 待斷詞的文字字串。

    回傳：
        token 清單（英數字詞 + 中文雙字詞）。
    """
    text = str(text)
    # 英數字詞：包含底線、#+. 等符號，保留程式語言名稱如 C++、C#。
    ascii_words = re.findall(r"[A-Za-z0-9_#+.]+", text)
    # 逐字擷取所有中文字元，再組成相鄰雙字 bigram。
    chinese_chars = re.findall(r"[一-鿿]", text)
    chinese_bigrams = ["".join(chinese_chars[i : i + 2]) for i in range(len(chinese_chars) - 1)]
    return [token for token in ascii_words + chinese_bigrams if len(token.strip()) > 1]


def build_tokenizer(token_counter: Counter):
    """
    建立用於 TfidfVectorizer 的斷詞函式，並同步記錄 token 頻率。

    優先使用 jieba 進行斷詞，jieba 未安裝時降級使用 fallback_chinese_tokenizer。
    每次斷詞結果都會更新 token_counter，供後續視覺化（Top tokens 圖）使用。

    參數：
        token_counter - 共用的 Counter 物件，用於統計全訓練集的 token 頻率。

    回傳：
        符合 TfidfVectorizer tokenizer 介面的函式。
    """
    def tokenizer(text: str) -> list[str]:
        if jieba is not None:
            # 使用 jieba 斷詞，過濾純標點與長度 <= 1 的 token。
            tokens = [
                word
                for word in jieba.cut(str(text))
                if len(word.strip()) > 1 and not re.match(r"^[^\w\s]+$", word)
            ]
        else:
            tokens = fallback_chinese_tokenizer(text)

        # 同步統計 token 頻率，供後續 Top tokens 視覺化使用。
        token_counter.update(tokens)
        return tokens

    return tokenizer


def add_text_clusters(
    df: pd.DataFrame,
    token_counter: Counter,
    n_clusters: int = TEXT_CLUSTER_COUNT,
    logger=None,
) -> pd.DataFrame:
    """
    對職缺文字進行 TF-IDF 向量化並以 KMeans 聚類，新增 job_cluster_label 欄位。

    流程：
    1. 合併 jobTitles + jobCategory 成 job_text_features 文字欄位
    2. 以 TF-IDF 將文字轉換為稀疏向量矩陣
    3. 以 KMeans 將職缺分成 n_clusters 群，結果存入 job_cluster_label

    資料筆數 < 2 時略過 KMeans，直接設 job_cluster_label='0'。

    參數：
        df            - 已補齊欄位的 DataFrame。
        token_counter - 用於統計 token 頻率的 Counter（傳給 build_tokenizer）。
        n_clusters    - KMeans 聚類群數，預設由 config.TEXT_CLUSTER_COUNT 控制。
        logger        - 可選的 Logger 物件。

    回傳：
        新增 job_text_features 與 job_cluster_label 欄位後的 DataFrame。
    """
    df = df.copy()
    # 清理 jobCategory 中的分隔符號（逗號、頓號、斜線、分號等），以空格取代。
    df["clean_category"] = df["jobCategory"].astype(str).str.replace("[,、/|;]+", " ", regex=True)
    df["clean_title"] = df["jobTitles"].astype(str)
    # 合併職缺標題與類別，作為文字特徵的輸入。
    df["job_text_features"] = df["clean_title"] + " " + df["clean_category"]
    if logger:
        logger.info("特徵工程：合併 jobTitles + jobCategory 成文字特徵 job_text_features")

    vectorizer = TfidfVectorizer(
        tokenizer=build_tokenizer(token_counter),
        max_features=TFIDF_MAX_FEATURES,
        lowercase=False,
        token_pattern=None,  # 使用自訂 tokenizer，不使用預設 token_pattern。
    )
    x_tfidf = vectorizer.fit_transform(df["job_text_features"])
    if logger:
        logger.info("特徵工程：使用 TF-IDF 轉換文字，矩陣大小=%s x %s", x_tfidf.shape[0], x_tfidf.shape[1])

    # 資料筆數不足 2 時無法執行 KMeans，直接指派群 0。
    if len(df) < 2:
        df["job_cluster_label"] = "0"
        if logger:
            logger.info("特徵工程：資料少於 2 筆，略過 KMeans，job_cluster_label 設為 0")
        return df

    # 群數不能超過資料筆數。
    cluster_count = min(n_clusters, len(df))
    df["job_cluster_label"] = KMeans(
        n_clusters=cluster_count,
        random_state=RANDOM_STATE,
        n_init=10,
    ).fit_predict(x_tfidf).astype(str)
    if logger:
        logger.info("特徵工程：使用 KMeans 建立職缺文字群集，群數=%s", cluster_count)
    return df


def build_model_frame(
    df: pd.DataFrame,
    salary_bound: str,
    logger=None,
) -> tuple[pd.DataFrame, pd.Series, list[str], Counter, pd.DataFrame]:
    """
    整合資料清洗、文字聚類與衍生特徵，產出完整的訓練用資料集。

    步驟：
    1. fill_required_columns  - 補齊缺值
    2. filter_tech_jobs       - 篩選科技職缺
    3. add_text_clusters      - 文字聚類（TF-IDF + KMeans）
    4. add_salary_target      - 解析薪資、新增目標欄位
    5. 衍生新特徵欄位：
       - clean_location           - 地區前三字（城市名稱）
       - employee_scale_cat       - 公司規模（字串類別）
       - announce_date_cat        - 職缺公告日期（字串類別）
       - experience_band          - 標準化年資區間
       - experience_band_cluster  - 年資區間 + 職缺群集的組合特徵
    6. 目標欄位套用 log1p，降低薪資分布的右偏態影響

    參數：
        df           - 讀取後的原始 DataFrame。
        salary_bound - 'max' 或 'min'，決定目標薪資取上限或下限。
        logger       - 可選的 Logger 物件。

    回傳：
        (x, y, feature_cols, token_counter, df_model) 五元組：
        - x             : 特徵 DataFrame
        - y             : 目標 Series（log1p 轉換後）
        - feature_cols  : 實際使用的特徵欄位名稱清單
        - token_counter : 全資料集的 token 頻率統計
        - df_model      : 含所有衍生欄位的完整 DataFrame（供視覺化使用）
    """
    df = fill_required_columns(df, logger=logger)
    df = filter_tech_jobs(df, logger=logger)
    token_counter: Counter = Counter()
    df = add_text_clusters(df, token_counter, logger=logger)
    df_model = add_salary_target(df, salary_bound, logger=logger)

    # 衍生特徵：地區取前三字元（去除縣市後的詳細地址）。
    df_model["clean_location"] = df_model["jobLocationAt"].astype(str).str.strip().str[:3]
    # 公司規模轉為字串類別，避免模型誤解為有序數值。
    df_model["employee_scale_cat"] = df_model["employeeCount"].astype(str)
    # 公告日期轉為字串類別，捕捉市場景氣的季節性特徵。
    df_model["announce_date_cat"] = df_model["jobAnnounceDate"].astype(str)
    # 年資標準化為固定區間字串。
    df_model["experience_band"] = df_model["jobRqYear"].apply(normalize_experience_band)
    # 組合特徵：年資區間 + 職缺文字群集，捕捉「資深後端」vs「初級前端」等交互效果。
    df_model["experience_band_cluster"] = (
        df_model["experience_band"].astype(str) + "_" + df_model["job_cluster_label"].astype(str)
    )

    # 定義候選特徵欄位清單（按重要性排序）。
    feature_candidates = [
        "keyword",
        "clean_location",
        "jobCompanyName",
        "jobCompanyIndustry",
        "employee_scale_cat",
        "experience_band",
        "jobRqDepartment",
        "job_cluster_label",
        "experience_band_cluster",
        "is_tech_job",
        "announce_date_cat",
    ]
    # 只保留實際存在於 df_model 的欄位（防止欄位缺失導致 KeyError）。
    feature_cols = [column for column in feature_candidates if column in df_model.columns]
    if logger:
        logger.info("特徵工程：建立衍生欄位 clean_location、employee_scale_cat、announce_date_cat、experience_band_cluster")
        logger.info("特徵工程：模型使用特徵欄位=%s", feature_cols)

    x = df_model[feature_cols].copy()
    # 將所有特徵欄位統一轉為字串，確保類別模型（CatBoost / LightGBM）可正確處理。
    for column in feature_cols:
        x[column] = x[column].astype(str)

    # 目標欄位套用 log1p 轉換，壓縮薪資的右偏分布，讓模型更容易學習。
    y = np.log1p(df_model[TARGET_COLUMN])
    if logger:
        logger.info("特徵工程：目標欄位 %s 套用 log1p，降低薪資偏態影響", TARGET_COLUMN)
    return x, y, feature_cols, token_counter, df_model


def split_data(x: pd.DataFrame, y: pd.Series, logger=None):
    """
    依 config.TEST_SIZE 比例切分訓練集與測試集。

    使用固定 RANDOM_STATE 確保每次切分結果一致，可重現。

    參數：
        x      - 特徵 DataFrame。
        y      - 目標 Series（log1p 薪資）。
        logger - 可選的 Logger 物件，用於記錄切分結果。

    回傳：
        (x_train, x_test, y_train, y_test) 四元組。
    """
    split = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    if logger:
        x_train, x_test, _, _ = split
        logger.info(
            "資料切分：train=%s 筆，test=%s 筆，test_size=%s，random_state=%s",
            len(x_train),
            len(x_test),
            TEST_SIZE,
            RANDOM_STATE,
        )
    return split


def ordinal_encode_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    以 OrdinalEncoder 對類別欄位進行整數編碼（供 XGBoost 使用）。

    Encoder 只在訓練集上 fit，測試集僅 transform。
    測試集中出現訓練集未見過的類別值，會以 -1 表示（unknown_value=-1）。

    參數：
        x_train - 訓練集特徵 DataFrame（全部為字串欄位）。
        x_test  - 測試集特徵 DataFrame。

    回傳：
        (x_train_encoded, x_test_encoded, encoded_columns) 三元組。
    """
    encoded_columns = list(x_train.columns)
    x_train_encoded = x_train.copy()
    x_test_encoded = x_test.copy()

    from sklearn.preprocessing import OrdinalEncoder

    # handle_unknown='use_encoded_value' + unknown_value=-1：
    # 測試集中未見過的類別值以 -1 填充，而非拋出例外。
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    x_train_encoded[encoded_columns] = encoder.fit_transform(x_train_encoded[encoded_columns].astype(str))
    x_test_encoded[encoded_columns] = encoder.transform(x_test_encoded[encoded_columns].astype(str))
    return x_train_encoded, x_test_encoded, encoded_columns


def align_categorical_columns(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    對齊兩個 DataFrame 中類別欄位的 categories 集合（供 LightGBM 使用）。

    LightGBM 要求訓練集與驗證/測試集的 Categorical 欄位擁有完全相同的 categories，
    否則預測時會拋出例外。此函式取兩個 DataFrame 各欄位的 union 作為共同 categories。

    參數：
        train_df     - 訓練集 DataFrame（Categorical 型別欄位）。
        other_df     - 驗證集或測試集 DataFrame。
        feature_cols - 需要對齊的欄位名稱清單。

    回傳：
        (train_df_aligned, other_df_aligned) 兩個對齊後的 DataFrame。
    """
    train_df = train_df.copy()
    other_df = other_df.copy()

    for column in feature_cols:
        # 取兩個 DataFrame 中該欄位所有出現過的類別值，取聯集作為共同 categories。
        categories = pd.Index(train_df[column].astype(str).unique()).union(other_df[column].astype(str).unique())
        train_df[column] = pd.Categorical(train_df[column].astype(str), categories=categories)
        other_df[column] = pd.Categorical(other_df[column].astype(str), categories=categories)

    return train_df, other_df
