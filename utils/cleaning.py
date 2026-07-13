"""
資料清洗工具模組。

負責將 104 人力銀行爬取的原始 CSV 資料，
清洗成可直接送入特徵工程與模型訓練的乾淨 DataFrame。

主要處理流程：
1. 讀取 CSV（自動判斷 UTF-8 / CP950 編碼）
2. 補齊必要欄位的缺值
3. 依科技關鍵字篩選出科技職缺
4. 解析薪資文字，轉換成月薪數值
5. 新增目標欄位 target_monthly_salary

所屬位置：Salary Predictive Analytics Engine/utils/cleaning.py
被以下模組引用：utils/engineering.py、main.py、catboost_main.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import TARGET_COLUMN


# 當欄位值缺失或無法判斷時使用的預設填充字串。
UNKNOWN_TEXT = "Unknown"

# 用於判斷「科技職缺」的關鍵字清單。
# 只要職缺標題（jobTitles）或職缺類別（jobCategory）包含以下任一關鍵字，
# 就會被標記為 is_tech_job=1。
TECH_KEYWORDS = [
    "AI",
    "Machine Learning",
    "ML",
    "Deep Learning",
    "Data",
    "資料",
    "數據",
    "軟體",
    "Software",
    "後端",
    "Backend",
    "前端",
    "Frontend",
    "全端",
    "Full Stack",
    "雲端",
    "Cloud",
    "DevOps",
    "SRE",
    "資安",
    "Security",
    "網路",
    "Network",
    "系統",
    "System",
    "資料庫",
    "Database",
    "演算法",
    "Algorithm",
    "韌體",
    "Firmware",
    "嵌入式",
    "產品經理",
    "產品管理師",
    "產品企劃",
    "軟體專案主管",
    "專案經理",
    "專案管理師",
    "其他專案管理師",
    "系統分析師",
    "系統規劃分析師",
    "Embedded",
    "半導體",
    "製程",
    "PM",
    "自動化",
    "測試",
    "QA",
    "工程師",
    "平台",
    "APP",
    "Web",
    "網站",
    "SaaS",
    "產品",
    "資訊",
    "IT",
    "數位",
    "電商",
    "後台",
    "前台",
    "API",
    "ERP",
    "CRM",
]

# 職缺標題中出現以下關鍵字，即使符合 TECH_KEYWORDS 也會被排除（非科技職缺）。
NON_TECH_TITLE_KEYWORDS = [
    "房仲",
    "不動產",
    "經紀人",
    "租車櫃台",
    "租車櫃檯",
    "iRent",
    "整備專員",
]

# 職缺類別中出現以下關鍵字，即使符合 TECH_KEYWORDS 也會被排除（非科技職缺）。
NON_TECH_CATEGORY_KEYWORDS = [
    "不動產經紀人",
    "不動產／商場開發人員",
    "經紀人",
    "門市／店員／專櫃人員",
    "店長／賣場管理人員",
    "國內業務",
    "通路開發人員",
    "專案業務主管",
    "工地監工／主任",
    "作業員／包裝員",
]


def read_104_csv(file_path: str | Path, logger=None) -> pd.DataFrame:
    """
    讀取 104 人力銀行匯出的 CSV 檔案。

    優先嘗試 UTF-8-sig 編碼（含 BOM），失敗時改用 CP950（Big5）。
    讀取成功後記錄資料筆數與欄位數。

    參數：
        file_path - CSV 檔案路徑。
        logger    - 可選的 Logger 物件，用於輸出讀取資訊。

    回傳：
        讀取完成的 DataFrame。
    """
    file_path = Path(file_path)
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        if logger:
            logger.info("讀取資料：%s，編碼=utf-8-sig，資料筆數=%s，欄位數=%s", file_path, df.shape[0], df.shape[1])
        return df
    except UnicodeDecodeError:
        # UTF-8 解碼失敗，改用 Big5（CP950）並忽略無法解碼的字元。
        df = pd.read_csv(file_path, encoding="cp950", encoding_errors="ignore")
        if logger:
            logger.info("讀取資料：%s，編碼=cp950，資料筆數=%s，欄位數=%s", file_path, df.shape[0], df.shape[1])
        return df


def fill_required_columns(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    補齊特徵工程與模型訓練所需的必要欄位缺值。

    若欄位不存在於 DataFrame 中，則新增該欄位並填入預設值。
    若欄位存在但有缺值（NaN），則以對應的預設值填充。
    數值欄位（employeeCount）以 0 填充，其餘文字欄位以 UNKNOWN_TEXT 填充。

    參數：
        df     - 原始 DataFrame。
        logger - 可選的 Logger 物件，用於記錄補值操作。

    回傳：
        補齊缺值後的新 DataFrame（不修改原始資料）。
    """
    df = df.copy()
    fill_values = {
        "jobTitles": UNKNOWN_TEXT,
        "jobCategory": UNKNOWN_TEXT,
        "jobRqYear": UNKNOWN_TEXT,
        "jobRqDepartment": UNKNOWN_TEXT,
        "jobCompanyName": UNKNOWN_TEXT,
        "jobCompanyIndustry": UNKNOWN_TEXT,
        "employeeCount": 0,
        "jobLocationAt": UNKNOWN_TEXT,
        "keyword": UNKNOWN_TEXT,
        "jobAnnounceDate": UNKNOWN_TEXT,
        "jobSalary": "",
        "jobSalaryType": "",
        "is_tech_job": 0,
    }

    for column, fill_value in fill_values.items():
        if column not in df.columns:
            # 欄位完全不存在，新增並填入預設值。
            df[column] = fill_value
            if logger:
                logger.info("清洗資料：新增缺少欄位 %s，預設值=%s", column, fill_value)
        else:
            missing_count = int(df[column].isna().sum())
            df[column] = df[column].fillna(fill_value)
            if logger and missing_count > 0:
                logger.info("清洗資料：欄位 %s 缺值補齊 %s 筆，補值=%s", column, missing_count, fill_value)

    return df


def _build_keyword_regex(keywords: list[str]) -> str:
    """
    將關鍵字清單編譯成單一正規表達式字串。

    純英數關鍵字加上單字邊界（\\b 效果），避免「IT」誤中「Twitter」等情況。
    含中文的關鍵字則直接匹配，不加邊界限制。

    參數：
        keywords - 關鍵字清單。

    回傳：
        可直接傳入 str.contains(..., regex=True) 的正規表達式字串。
    """
    pattern_parts = []
    for keyword in keywords:
        escaped_keyword = re.escape(keyword).replace(r"\ ", r"\s+")
        if re.fullmatch(r"[A-Za-z0-9+#.\s]+", keyword):
            # 純英數關鍵字：在前後加上「非英數」邊界，避免誤匹配子字串。
            pattern_parts.append(rf"(?<![A-Za-z0-9]){escaped_keyword}(?![A-Za-z0-9])")
        else:
            # 含中文或特殊字元的關鍵字：直接匹配，不加邊界。
            pattern_parts.append(escaped_keyword)
    return "|".join(pattern_parts)


def filter_tech_jobs(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    標記科技職缺並過濾出 is_tech_job=1 的資料列。

    Step A：若 jobCategory 或 jobTitles 包含 TECH_KEYWORDS 中任一關鍵字，
            則將該列標記為 is_tech_job=1。
    Step B：若同時符合 NON_TECH_TITLE_KEYWORDS 或 NON_TECH_CATEGORY_KEYWORDS，
            則將 is_tech_job 強制設回 0，排除非科技職缺（如不動產、租車等）。

    最終只保留 is_tech_job=1 的資料列。

    參數：
        df     - 已補齊欄位的 DataFrame。
        logger - 可選的 Logger 物件，用於記錄篩選統計。

    回傳：
        只含科技職缺的新 DataFrame。
    """
    df = df.copy()
    before_count = len(df)

    # 預先編譯正規表達式，避免在 apply 迴圈內重複編譯。
    tech_pattern = _build_keyword_regex(TECH_KEYWORDS)
    non_tech_title_pattern = _build_keyword_regex(NON_TECH_TITLE_KEYWORDS)
    non_tech_category_pattern = _build_keyword_regex(NON_TECH_CATEGORY_KEYWORDS)

    category_text = df["jobCategory"].fillna("").astype(str)
    title_text = df["jobTitles"].fillna("").astype(str)

    # Step A：在 jobCategory 或 jobTitles 中找到科技關鍵字 → is_tech_job=1。
    tech_category_match = category_text.str.contains(tech_pattern, case=False, na=False, regex=True)
    tech_title_match = title_text.str.contains(tech_pattern, case=False, na=False, regex=True)
    df["is_tech_job"] = (tech_category_match | tech_title_match).astype(int)

    # Step B：符合非科技職缺關鍵字的列，強制將 is_tech_job 設回 0。
    non_tech_title_match = title_text.str.contains(non_tech_title_pattern, case=False, na=False, regex=True)
    non_tech_category_match = category_text.str.contains(non_tech_category_pattern, case=False, na=False, regex=True)
    non_tech_match = non_tech_title_match | non_tech_category_match

    matched_count = int(df["is_tech_job"].sum())
    excluded_count = int(((df["is_tech_job"] == 1) & non_tech_match).sum())
    df.loc[non_tech_match, "is_tech_job"] = 0
    df_filtered = df[df["is_tech_job"] == 1].copy()

    if logger:
        logger.info(
            "清洗資料 Step A：依科技關鍵字標記 is_tech_job=1，命中=%s/%s 筆",
            matched_count,
            before_count,
        )
        logger.info(
            "清洗資料 Step B：排除非 tech 職缺 %s 筆，只保留 is_tech_job=1，剩餘=%s 筆",
            excluded_count,
            len(df_filtered),
        )

    return df_filtered


def extract_salary_numbers(salary_text: str) -> list[int]:
    """
    從薪資字串中擷取所有數字（處理千位逗號分隔）。

    例如：'40,000~55,000' → [40000, 55000]
         '年薪 120萬' → [120]（萬字不含，需後續解析）

    參數：
        salary_text - 原始薪資文字字串。

    回傳：
        擷取到的整數清單，無數字時回傳空清單。
    """
    return [int(number.replace(",", "")) for number in re.findall(r"\d[\d,]*", str(salary_text))]


def _is_unusable_salary(salary_text: str, salary_type: str) -> bool:
    """
    判斷薪資字串是否為「無法使用」的薪資（如面議、論件）。

    面議薪資無法轉換成數值，必須排除。

    參數：
        salary_text - jobSalary 欄位文字。
        salary_type - jobSalaryType 欄位文字。

    回傳：
        True 表示此薪資無法使用，應排除。
    """
    combined = f"{salary_text} {salary_type}"
    unusable_keywords = ["面議", "論件", "按件", "待遇面議"]
    return any(keyword in combined for keyword in unusable_keywords)


def parse_monthly_salary(row: pd.Series, salary_bound: str) -> float:
    """
    將單一資料列的薪資欄位解析為月薪數值。

    解析邏輯：
    - 面議、論件計酬 → 排除（回傳 NaN）
    - 時薪、日薪 → 排除（回傳 NaN）
    - salary_bound='max' → 取薪資範圍最大值，'min' → 取最小值
    - 年薪 → 除以 12 換算月薪
    - 月薪 → 直接使用
    - 其他（無單位）→ 數值 < 300,000 視為月薪，否則視為年薪除以 12
    - 月薪範圍外（< 20,000 或 > 500,000）→ 排除（回傳 NaN）

    參數：
        row          - 單一 DataFrame 資料列，需含 jobSalary、jobSalaryType 欄位。
        salary_bound - 'max' 取上限薪資，'min' 取下限薪資。

    回傳：
        月薪數值（float），無法解析時回傳 NaN。
    """
    salary_text = str(row.get("jobSalary", ""))
    salary_type = str(row.get("jobSalaryType", ""))
    numbers = extract_salary_numbers(salary_text)

    if not numbers or _is_unusable_salary(salary_text, salary_type):
        return np.nan

    # 時薪與日薪無法直接換算月薪，一律排除。
    if "時薪" in salary_text or "時薪" in salary_type or "日薪" in salary_text or "日薪" in salary_type:
        return np.nan

    # 依 salary_bound 取薪資範圍的最大值或最小值。
    salary = max(numbers) if salary_bound == "max" else min(numbers)
    combined = f"{salary_text} {salary_type}"

    if "年薪" in combined:
        # 年薪除以 12 換算月薪。
        monthly_salary = salary / 12
    elif "月薪" in combined:
        monthly_salary = salary
    else:
        # 無明確單位：數值 >= 300,000 很可能是年薪，否則視為月薪。
        monthly_salary = salary if salary < 300_000 else salary / 12

    # 合理月薪範圍：20,000 ~ 500,000，超出範圍視為異常資料排除。
    if monthly_salary < 20_000 or monthly_salary > 500_000:
        return np.nan
    return float(monthly_salary)


def add_salary_target(df: pd.DataFrame, salary_bound: str, logger=None) -> pd.DataFrame:
    """
    為 DataFrame 新增目標欄位 target_monthly_salary。

    對每一列呼叫 parse_monthly_salary 解析薪資，
    無法解析的列（NaN）會被移除，不納入訓練集。

    參數：
        df           - 已篩選過科技職缺的 DataFrame。
        salary_bound - 'max' 使用薪資上限，'min' 使用薪資下限。
        logger       - 可選的 Logger 物件，用於記錄資料筆數變化。

    回傳：
        新增目標欄位且移除無效薪資列後的 DataFrame。
    """
    if salary_bound not in {"max", "min"}:
        raise ValueError("salary_bound must be 'max' or 'min'")

    df = df.copy()
    before_count = len(df)
    df[TARGET_COLUMN] = df.apply(parse_monthly_salary, axis=1, salary_bound=salary_bound)
    # 移除薪資無法解析的列（NaN），只保留有有效月薪的資料。
    df_model = df.dropna(subset=[TARGET_COLUMN]).copy()
    if logger:
        logger.info(
            "清洗資料：以薪資%s值建立目標欄位 %s，可用訓練資料=%s/%s，移除=%s 筆",
            salary_bound,
            TARGET_COLUMN,
            len(df_model),
            before_count,
            before_count - len(df_model),
        )
    return df_model
