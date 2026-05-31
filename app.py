# ===============================================================
# КУРСОВА РОБОТА
# Тема: Аналіз впливу характеристик смартфонів на їхню ринкову ціну
# Streamlit-застосунок
# ===============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import math


# ===============================================================
# 1. НАЛАШТУВАННЯ СТОРІНКИ
# ===============================================================

st.set_page_config(
    page_title="Аналіз ціни смартфонів",
    page_icon="📱",
    layout="wide"
)


# ===============================================================
# 2. ВЛАСНОРУЧНІ СТАТИСТИЧНІ ФУНКЦІЇ
# ===============================================================

def is_missing(value):
    if value is None:
        return True
    try:
        return math.isnan(value)
    except Exception:
        return False


def manual_len(values):
    count = 0
    for _ in values:
        count += 1
    return count


def manual_sum(values):
    total = 0.0
    for value in values:
        total += value
    return total


def clean_numeric_list(values):
    result = []
    for value in values:
        if not is_missing(value):
            result.append(float(value))
    return result


def mean_manual(values):
    values = clean_numeric_list(values)
    n = manual_len(values)
    if n == 0:
        return None
    return manual_sum(values) / n


def quick_sort(values):
    values = list(values)
    if manual_len(values) <= 1:
        return values

    pivot = values[manual_len(values) // 2]
    left = []
    middle = []
    right = []

    for value in values:
        if value < pivot:
            left.append(value)
        elif value > pivot:
            right.append(value)
        else:
            middle.append(value)

    return quick_sort(left) + middle + quick_sort(right)


def median_manual(values):
    values = clean_numeric_list(values)
    n = manual_len(values)

    if n == 0:
        return None

    values = quick_sort(values)
    middle = n // 2

    if n % 2 == 1:
        return values[middle]

    return (values[middle - 1] + values[middle]) / 2


def variance_manual(values):
    values = clean_numeric_list(values)
    n = manual_len(values)

    if n == 0:
        return None

    avg = mean_manual(values)
    total = 0.0

    for value in values:
        total += (value - avg) ** 2

    return total / n


def std_manual(values):
    var = variance_manual(values)

    if var is None:
        return None

    return math.sqrt(var)


def min_manual(values):
    values = clean_numeric_list(values)

    if manual_len(values) == 0:
        return None

    current = values[0]

    for value in values:
        if value < current:
            current = value

    return current


def max_manual(values):
    values = clean_numeric_list(values)

    if manual_len(values) == 0:
        return None

    current = values[0]

    for value in values:
        if value > current:
            current = value

    return current


def quartiles_manual(values):
    values = quick_sort(clean_numeric_list(values))
    n = manual_len(values)

    if n == 0:
        return None, None, None

    middle = n // 2

    if n % 2 == 0:
        lower_half = values[:middle]
        upper_half = values[middle:]
    else:
        lower_half = values[:middle]
        upper_half = values[middle + 1:]

    q1 = median_manual(lower_half)
    q2 = median_manual(values)
    q3 = median_manual(upper_half)

    return q1, q2, q3


def iqr_bounds_manual(values):
    q1, _, q3 = quartiles_manual(values)

    if q1 is None or q3 is None:
        return None, None

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return lower, upper


def pearson_manual(x_values, y_values):
    pairs = []

    for x, y in zip(x_values, y_values):
        if not is_missing(x) and not is_missing(y):
            pairs.append((float(x), float(y)))

    if manual_len(pairs) == 0:
        return None

    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]

    x_mean = mean_manual(xs)
    y_mean = mean_manual(ys)

    numerator = 0.0
    x_denominator = 0.0
    y_denominator = 0.0

    for x, y in pairs:
        numerator += (x - x_mean) * (y - y_mean)
        x_denominator += (x - x_mean) ** 2
        y_denominator += (y - y_mean) ** 2

    denominator = math.sqrt(x_denominator * y_denominator)

    if denominator == 0:
        return 0

    return numerator / denominator


def mae_manual(y_true, y_pred):
    total = 0.0
    n = manual_len(y_true)

    for real, pred in zip(y_true, y_pred):
        total += abs(real - pred)

    return total / n


def rmse_manual(y_true, y_pred):
    total = 0.0
    n = manual_len(y_true)

    for real, pred in zip(y_true, y_pred):
        total += (real - pred) ** 2

    return math.sqrt(total / n)


def r2_manual(y_true, y_pred):
    avg = mean_manual(y_true)

    ss_res = 0.0
    ss_tot = 0.0

    for real, pred in zip(y_true, y_pred):
        ss_res += (real - pred) ** 2
        ss_tot += (real - avg) ** 2

    if ss_tot == 0:
        return 0

    return 1 - ss_res / ss_tot


# ===============================================================
# 3. МАТРИЧНІ ОПЕРАЦІЇ ДЛЯ МНК
# ===============================================================

def transpose_matrix(matrix):
    return [
        [matrix[i][j] for i in range(manual_len(matrix))]
        for j in range(manual_len(matrix[0]))
    ]


def multiply_matrix(a, b):
    result = []

    for i in range(manual_len(a)):
        row = []

        for j in range(manual_len(b[0])):
            total = 0.0

            for k in range(manual_len(b)):
                total += a[i][k] * b[k][j]

            row.append(total)

        result.append(row)

    return result


def inverse_matrix(matrix):
    n = manual_len(matrix)

    augmented = []

    for i in range(n):
        row = []

        for j in range(n):
            row.append(float(matrix[i][j]))

        for j in range(n):
            row.append(1.0 if i == j else 0.0)

        augmented.append(row)

    for col in range(n):
        pivot_row = col

        for row in range(col + 1, n):
            if abs(augmented[row][col]) > abs(augmented[pivot_row][col]):
                pivot_row = row

        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]

        if abs(pivot) < 1e-12:
            raise ValueError("Матрицю неможливо обернути. Ймовірно, деякі ознаки лінійно залежні.")

        for j in range(2 * n):
            augmented[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = augmented[row][col]

                for j in range(2 * n):
                    augmented[row][j] -= factor * augmented[col][j]

    inverse = []

    for i in range(n):
        inverse.append(augmented[i][n:])

    return inverse


def linear_regression_mnk_manual(x_matrix, y_values):
    y_matrix = [[float(y)] for y in y_values]

    xt = transpose_matrix(x_matrix)
    xtx = multiply_matrix(xt, x_matrix)
    xtx_inv = inverse_matrix(xtx)
    xty = multiply_matrix(xt, y_matrix)
    beta_matrix = multiply_matrix(xtx_inv, xty)

    return [row[0] for row in beta_matrix]


def predict_manual(x_matrix, beta):
    predictions = []

    for row in x_matrix:
        value = 0.0

        for j in range(manual_len(beta)):
            value += row[j] * beta[j]

        predictions.append(value)

    return predictions


# ===============================================================
# 4. ПАРСИНГ ТЕКСТОВИХ ХАРАКТЕРИСТИК
# ===============================================================

def normalize_text(text):
    return str(text).replace("\u2009", " ").replace("\xa0", " ").strip()


def extract_first_number(text, pattern, default=None):
    if pd.isna(text):
        return default

    text = normalize_text(text)
    match = re.search(pattern, text, flags=re.IGNORECASE)

    if match:
        return float(match.group(1))

    return default


def parse_price(text):
    text = normalize_text(text)
    text = text.replace("₹", "").replace(",", "")

    match = re.search(r"(\d+(?:\.\d+)?)", text)

    if match:
        return float(match.group(1))

    return None


def parse_ram(text):
    return extract_first_number(text, r"(\d+(?:\.\d+)?)\s*GB\s*RAM")


def parse_storage(text):
    text = normalize_text(text)

    tb_match = re.search(r"(\d+(?:\.\d+)?)\s*TB\s*inbuilt", text, flags=re.IGNORECASE)
    if tb_match:
        return float(tb_match.group(1)) * 1024

    gb_match = re.search(r"(\d+(?:\.\d+)?)\s*GB\s*inbuilt", text, flags=re.IGNORECASE)
    if gb_match:
        return float(gb_match.group(1))

    mb_match = re.search(r"(\d+(?:\.\d+)?)\s*MB\s*inbuilt", text, flags=re.IGNORECASE)
    if mb_match:
        return float(mb_match.group(1)) / 1024

    return None


def parse_battery(text):
    return extract_first_number(text, r"(\d+(?:\.\d+)?)\s*mAh")


def parse_fast_charging(text):
    value = extract_first_number(text, r"(\d+(?:\.\d+)?)\s*W", default=0)

    if value is None:
        return 0

    return value


def parse_screen_size(text):
    return extract_first_number(text, r"(\d+(?:\.\d+)?)\s*inches")


def parse_refresh_rate(text):
    value = extract_first_number(text, r"(\d+(?:\.\d+)?)\s*Hz", default=60)

    if value is None:
        return 60

    return value


def parse_rear_camera(text):
    return extract_first_number(text, r"(\d+(?:\.\d+)?)\s*MP")


def parse_front_camera(text):
    return extract_first_number(text, r"&\s*(\d+(?:\.\d+)?)\s*MP\s*Front")


def parse_processor_ghz(text):
    return extract_first_number(text, r"(\d+(?:\.\d+)?)\s*GHz")


def parse_cores(text):
    text = normalize_text(text).lower()

    if "octa" in text:
        return 8
    if "hexa" in text:
        return 6
    if "quad" in text:
        return 4
    if "dual" in text:
        return 2

    return None


def parse_has_5g(text):
    return 1 if "5G" in normalize_text(text) else 0


def parse_has_nfc(text):
    return 1 if "NFC" in normalize_text(text) else 0


def parse_is_ios(text):
    return 1 if "ios" in normalize_text(text).lower() else 0


def parse_supports_card(text):
    text = normalize_text(text)

    if "Not Supported" in text:
        return 0

    return 1


# ===============================================================
# 5. ОБРОБКА ДАНИХ
# ===============================================================

def process_dataset(raw_df):
    required_columns = [
        "model", "price", "rating", "sim", "processor",
        "ram", "battery", "display", "camera", "card", "os"
    ]

    missing_columns = []

    for column in required_columns:
        if column not in raw_df.columns:
            missing_columns.append(column)

    if manual_len(missing_columns) > 0:
        raise ValueError("У файлі відсутні потрібні колонки: " + str(missing_columns))

    clean_df = pd.DataFrame()

    clean_df["model"] = raw_df["model"]
    clean_df["price"] = raw_df["price"].apply(parse_price)
    clean_df["rating"] = pd.to_numeric(raw_df["rating"], errors="coerce")

    clean_df["ram_gb"] = raw_df["ram"].apply(parse_ram)
    clean_df["storage_gb"] = raw_df["ram"].apply(parse_storage)

    clean_df["battery_mah"] = raw_df["battery"].apply(parse_battery)
    clean_df["fast_charging_w"] = raw_df["battery"].apply(parse_fast_charging)

    clean_df["screen_size_in"] = raw_df["display"].apply(parse_screen_size)
    clean_df["refresh_rate_hz"] = raw_df["display"].apply(parse_refresh_rate)

    clean_df["rear_camera_mp"] = raw_df["camera"].apply(parse_rear_camera)
    clean_df["front_camera_mp"] = raw_df["camera"].apply(parse_front_camera)

    clean_df["processor_ghz"] = raw_df["processor"].apply(parse_processor_ghz)
    clean_df["cores"] = raw_df["processor"].apply(parse_cores)

    clean_df["has_5g"] = raw_df["sim"].apply(parse_has_5g)
    clean_df["has_nfc"] = raw_df["sim"].apply(parse_has_nfc)
    clean_df["is_ios"] = raw_df["os"].apply(parse_is_ios)
    clean_df["supports_card"] = raw_df["card"].apply(parse_supports_card)

    features = [
        "rating",
        "ram_gb",
        "storage_gb",
        "battery_mah",
        "fast_charging_w",
        "screen_size_in",
        "refresh_rate_hz",
        "rear_camera_mp",
        "front_camera_mp",
        "processor_ghz",
        "cores",
        "has_5g",
        "has_nfc",
        "is_ios",
        "supports_card"
    ]

    before_dropna = len(clean_df)
    model_df = clean_df[["price"] + features].dropna().copy()
    after_dropna = len(model_df)

    lower_price, upper_price = iqr_bounds_manual(model_df["price"].tolist())

    before_iqr = len(model_df)
    model_df = model_df[
        (model_df["price"] >= lower_price) &
        (model_df["price"] <= upper_price)
    ].copy()
    after_iqr = len(model_df)

    info = {
        "features": features,
        "before_dropna": before_dropna,
        "after_dropna": after_dropna,
        "removed_missing": before_dropna - after_dropna,
        "lower_price": lower_price,
        "upper_price": upper_price,
        "before_iqr": before_iqr,
        "after_iqr": after_iqr,
        "removed_outliers": before_iqr - after_iqr
    }

    return clean_df, model_df, info


def build_statistics_table(model_df, features):
    rows = []

    for column in ["price"] + features:
        values = model_df[column].tolist()
        q1, median_value, q3 = quartiles_manual(values)

        rows.append({
            "Ознака": column,
            "Кількість": manual_len(values),
            "Мінімум": round(min_manual(values), 3),
            "Максимум": round(max_manual(values), 3),
            "Середнє": round(mean_manual(values), 3),
            "Медіана": round(median_value, 3),
            "Станд. відхилення": round(std_manual(values), 3),
            "Q1": round(q1, 3),
            "Q3": round(q3, 3)
        })

    return pd.DataFrame(rows)


def build_correlation_table(model_df, features):
    rows = []

    for feature in features:
        r = pearson_manual(model_df[feature].tolist(), model_df["price"].tolist())

        rows.append({
            "Ознака": feature,
            "Кореляція з ціною": r,
            "Модуль кореляції": abs(r)
        })

    result = pd.DataFrame(rows)
    result = result.sort_values(by="Модуль кореляції", ascending=False)

    return result


def train_model(model_df, features):
    all_indices = list(range(len(model_df)))

    train_indices = []
    test_indices = []

    for position, index in enumerate(all_indices):
        if position % 5 == 0:
            test_indices.append(index)
        else:
            train_indices.append(index)

    train_df = model_df.iloc[train_indices].copy()
    test_df = model_df.iloc[test_indices].copy()

    feature_means = {}
    feature_stds = {}

    for feature in features:
        feature_means[feature] = mean_manual(train_df[feature].tolist())
        feature_stds[feature] = std_manual(train_df[feature].tolist())

        if feature_stds[feature] == 0:
            feature_stds[feature] = 1

    def make_regression_matrix(df_part):
        matrix = []

        for _, row in df_part.iterrows():
            values = [1.0]

            for feature in features:
                standardized = (row[feature] - feature_means[feature]) / feature_stds[feature]
                values.append(float(standardized))

            matrix.append(values)

        return matrix

    x_train = make_regression_matrix(train_df)
    y_train = train_df["price"].tolist()

    x_test = make_regression_matrix(test_df)
    y_test = test_df["price"].tolist()

    beta = linear_regression_mnk_manual(x_train, y_train)
    y_pred = predict_manual(x_test, beta)

    mae = mae_manual(y_test, y_pred)
    rmse = rmse_manual(y_test, y_pred)
    r2 = r2_manual(y_test, y_pred)

    coef_rows = []

    for feature, coef in zip(features, beta[1:]):
        coef_rows.append({
            "Ознака": feature,
            "Стандартизований коефіцієнт": coef,
            "Модуль коефіцієнта": abs(coef)
        })

    coef_df = pd.DataFrame(coef_rows)
    coef_df = coef_df.sort_values(by="Модуль коефіцієнта", ascending=False)

    model_info = {
        "train_df": train_df,
        "test_df": test_df,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "beta": beta,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "coef_df": coef_df
    }

    return model_info


def predict_new_phone(new_phone, features, feature_means, feature_stds, beta):
    new_row = [1.0]

    for feature in features:
        standardized = (new_phone[feature] - feature_means[feature]) / feature_stds[feature]
        new_row.append(standardized)

    return predict_manual([new_row], beta)[0]


# ===============================================================
# 6. ГРАФІКИ
# ===============================================================

def plot_hist_price(model_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(model_df["price"], bins=30)
    ax.set_title("Розподіл ринкових цін смартфонів")
    ax.set_xlabel("Ціна, INR")
    ax.set_ylabel("Кількість моделей")
    ax.grid(True)
    return fig


def plot_box_price(model_df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(model_df["price"], vert=False)
    ax.set_title("Boxplot цін смартфонів після IQR-очищення")
    ax.set_xlabel("Ціна, INR")
    ax.grid(True)
    return fig


def plot_scatter(model_df, column, xlabel, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(model_df[column], model_df["price"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ціна, INR")
    ax.grid(True)
    return fig


def plot_correlation_bar(corr_df):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(corr_df["Ознака"], corr_df["Кореляція з ціною"])
    ax.set_title("Кореляція характеристик смартфона з ринковою ціною")
    ax.set_xlabel("Коефіцієнт кореляції Пірсона")
    ax.grid(True)
    ax.invert_yaxis()
    return fig


def plot_heatmap(model_df):
    heatmap_columns = [
        "price",
        "rating",
        "ram_gb",
        "storage_gb",
        "processor_ghz",
        "fast_charging_w",
        "refresh_rate_hz",
        "front_camera_mp",
        "has_5g",
        "has_nfc"
    ]

    heatmap_matrix = []

    for row_column in heatmap_columns:
        row = []

        for col_column in heatmap_columns:
            value = pearson_manual(
                model_df[col_column].tolist(),
                model_df[row_column].tolist()
            )
            row.append(value)

        heatmap_matrix.append(row)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(heatmap_matrix, vmin=-1, vmax=1)

    fig.colorbar(image, ax=ax, label="Коефіцієнт кореляції")

    ax.set_xticks(range(len(heatmap_columns)))
    ax.set_xticklabels(heatmap_columns, rotation=45, ha="right")

    ax.set_yticks(range(len(heatmap_columns)))
    ax.set_yticklabels(heatmap_columns)

    ax.set_title("Heatmap кореляцій між характеристиками смартфонів")

    for i in range(len(heatmap_columns)):
        for j in range(len(heatmap_columns)):
            ax.text(j, i, f"{heatmap_matrix[i][j]:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(y_test, y_pred)

    min_value = min(min(y_test), min(y_pred))
    max_value = max(max(y_test), max(y_pred))

    ax.plot([min_value, max_value], [min_value, max_value])

    ax.set_title("Фактична ціна vs прогнозована ціна")
    ax.set_xlabel("Фактична ціна, INR")
    ax.set_ylabel("Прогнозована ціна, INR")
    ax.grid(True)

    return fig


# ===============================================================
# 7. ІНТЕРФЕЙС STREAMLIT
# ===============================================================

st.title("📱 Аналіз впливу характеристик смартфонів на їхню ринкову ціну")

st.markdown(
    """
    Цей програмний застосунок створено для курсової роботи з аналітики даних.
    Він дозволяє завантажити CSV-файл з характеристиками смартфонів,
    очистити дані, виконати статистичний аналіз, побудувати графіки,
    визначити кореляції та спрогнозувати ринкову ціну смартфона.
    """
)

with st.expander("ℹ️ Інформація про роботу", expanded=True):
    st.write("**Тема:** Аналіз впливу характеристик смартфонів на їхню ринкову ціну")
    st.write("**Мета:** дослідити вплив характеристик смартфонів на їхню ринкову ціну та побудувати модель прогнозування вартості.")
    st.write("**Об'єкт дослідження:** ринок смартфонів, представлений набором даних з характеристиками та цінами пристроїв.")
    st.write("**Предмет дослідження:** залежність ринкової ціни смартфонів від технічних і функціональних характеристик.")
    st.write("**Методи:** описова статистика, IQR, EDA, кореляція Пірсона, множинна лінійна регресія, МНК, MAE, RMSE, R².")
    st.write("**Особливість реалізації:** основні методи реалізовано власноруч без використання sklearn.")


uploaded_file = st.file_uploader("Завантажте CSV-файл SmartPhones Dataset", type=["csv"])

if uploaded_file is None:
    st.warning("Завантажте CSV-файл, щоб почати аналіз.")
    st.stop()


try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Не вдалося прочитати CSV-файл: {error}")
    st.stop()


st.success("CSV-файл успішно завантажено.")

st.subheader("1. Початковий набір даних")

col1, col2 = st.columns(2)

with col1:
    st.metric("Кількість рядків", raw_df.shape[0])

with col2:
    st.metric("Кількість стовпців", raw_df.shape[1])

st.dataframe(raw_df.head(), use_container_width=True)

required_columns = [
    "model", "price", "rating", "sim", "processor",
    "ram", "battery", "display", "camera", "card", "os"
]

missing_report = raw_df[required_columns].isna().sum().reset_index()
missing_report.columns = ["Стовпець", "Кількість пропусків"]

st.subheader("2. Перевірка пропущених значень")
st.dataframe(missing_report, use_container_width=True)


try:
    clean_df, model_df, info = process_dataset(raw_df)
except Exception as error:
    st.error(f"Помилка під час обробки даних: {error}")
    st.stop()


features = info["features"]

st.subheader("3. Обробка та очищення даних")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Початково записів", raw_df.shape[0])

with col2:
    st.metric("Після видалення пропусків", info["after_dropna"])

with col3:
    st.metric("Після IQR-очищення", info["after_iqr"])

st.write(f"Вилучено через пропуски: **{info['removed_missing']}**")
st.write(f"Вилучено цінових викидів: **{info['removed_outliers']}**")
st.write(f"Нижня межа IQR для ціни: **{round(info['lower_price'], 2)}**")
st.write(f"Верхня межа IQR для ціни: **{round(info['upper_price'], 2)}**")

st.subheader("4. Сформовані числові ознаки")
st.dataframe(clean_df.head(), use_container_width=True)

clean_csv = clean_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Завантажити очищений CSV",
    data=clean_csv,
    file_name="smartphones_cleaned_features.csv",
    mime="text/csv"
)


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Описова статистика",
        "Графіки",
        "Кореляції",
        "Модель",
        "Прогноз"
    ]
)


with tab1:
    st.subheader("Описова статистика очищеного набору даних")

    stats_df = build_statistics_table(model_df, features)
    st.dataframe(stats_df, use_container_width=True)

    mean_price = mean_manual(model_df["price"].tolist())
    median_price = median_manual(model_df["price"].tolist())

    st.info(
        f"Середня ціна смартфона становить {round(mean_price, 2)} INR, "
        f"а медіанна ціна — {round(median_price, 2)} INR. "
        f"Якщо середнє більше за медіану, це свідчить про правий хвіст розподілу цін."
    )


with tab2:
    st.subheader("Графічний розвідувальний аналіз")

    st.pyplot(plot_hist_price(model_df))
    st.write("Більшість смартфонів зосереджена у бюджетному та середньому цінових сегментах.")

    st.pyplot(plot_box_price(model_df))
    st.write("Boxplot показує, що після IQR-очищення найсильніші цінові викиди було вилучено.")

    st.pyplot(plot_scatter(model_df, "ram_gb", "RAM, GB", "Залежність ціни від оперативної пам'яті"))
    st.pyplot(plot_scatter(model_df, "storage_gb", "Storage, GB", "Залежність ціни від внутрішньої пам'яті"))
    st.pyplot(plot_scatter(model_df, "fast_charging_w", "Fast charging, W", "Залежність ціни від швидкої зарядки"))


with tab3:
    st.subheader("Кореляційний аналіз")

    corr_df = build_correlation_table(model_df, features)

    st.dataframe(
        corr_df[["Ознака", "Кореляція з ціною"]],
        use_container_width=True
    )

    st.pyplot(plot_correlation_bar(corr_df))

    st.subheader("Heatmap кореляцій")
    st.pyplot(plot_heatmap(model_df))

    top_features = corr_df.head(8)

    st.info(
        "Найсильніше з ціною пов'язані такі характеристики: "
        + ", ".join(top_features["Ознака"].tolist())
        + "."
    )


with tab4:
    st.subheader("Модель множинної лінійної регресії")

    try:
        model_info = train_model(model_df, features)
    except Exception as error:
        st.error(f"Не вдалося побудувати модель: {error}")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", round(model_info["mae"], 2))

    with col2:
        st.metric("RMSE", round(model_info["rmse"], 2))

    with col3:
        st.metric("R²", round(model_info["r2"], 3))

    st.write(
        f"Коефіцієнт детермінації R² = **{round(model_info['r2'], 3)}**, "
        f"тобто модель пояснює приблизно **{round(model_info['r2'] * 100, 1)}%** варіації ціни смартфонів."
    )

    st.subheader("Коефіцієнти моделі")
    st.dataframe(
        model_info["coef_df"][["Ознака", "Стандартизований коефіцієнт"]],
        use_container_width=True
    )

    st.subheader("Фактична ціна vs прогнозована ціна")
    st.pyplot(plot_actual_vs_predicted(model_info["y_test"], model_info["y_pred"]))

    st.info(
        "Модель відображає загальну тенденцію ціноутворення, але не може повністю врахувати "
        "бренд, рік випуску, маркетинг, регіон продажу та інші ринкові фактори."
    )


with tab5:
    st.subheader("Прогнозування ціни нового смартфона")

    model_info = train_model(model_df, features)

    col1, col2, col3 = st.columns(3)

    with col1:
        rating = st.number_input("Рейтинг", min_value=0.0, max_value=100.0, value=85.0)
        ram_gb = st.number_input("RAM, GB", min_value=0.0, value=8.0)
        storage_gb = st.number_input("Внутрішня пам'ять, GB", min_value=0.0, value=256.0)
        battery_mah = st.number_input("Акумулятор, mAh", min_value=0.0, value=5000.0)
        fast_charging_w = st.number_input("Швидка зарядка, W", min_value=0.0, value=67.0)

    with col2:
        screen_size_in = st.number_input("Діагональ екрана, inches", min_value=0.0, value=6.7)
        refresh_rate_hz = st.number_input("Частота оновлення, Hz", min_value=0.0, value=120.0)
        rear_camera_mp = st.number_input("Основна камера, MP", min_value=0.0, value=50.0)
        front_camera_mp = st.number_input("Фронтальна камера, MP", min_value=0.0, value=32.0)
        processor_ghz = st.number_input("Частота процесора, GHz", min_value=0.0, value=2.8)

    with col3:
        cores = st.number_input("Кількість ядер", min_value=1, value=8)
        has_5g = st.selectbox("Підтримка 5G", [1, 0], format_func=lambda x: "Так" if x == 1 else "Ні")
        has_nfc = st.selectbox("Наявність NFC", [1, 0], format_func=lambda x: "Так" if x == 1 else "Ні")
        is_ios = st.selectbox("Операційна система iOS", [0, 1], format_func=lambda x: "Так" if x == 1 else "Ні")
        supports_card = st.selectbox("Підтримка карти пам'яті", [1, 0], format_func=lambda x: "Так" if x == 1 else "Ні")

    new_phone = {
        "rating": rating,
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "battery_mah": battery_mah,
        "fast_charging_w": fast_charging_w,
        "screen_size_in": screen_size_in,
        "refresh_rate_hz": refresh_rate_hz,
        "rear_camera_mp": rear_camera_mp,
        "front_camera_mp": front_camera_mp,
        "processor_ghz": processor_ghz,
        "cores": cores,
        "has_5g": has_5g,
        "has_nfc": has_nfc,
        "is_ios": is_ios,
        "supports_card": supports_card
    }

    if st.button("Спрогнозувати ціну"):
        predicted_price = predict_new_phone(
            new_phone,
            features,
            model_info["feature_means"],
            model_info["feature_stds"],
            model_info["beta"]
        )

        st.success(f"Прогнозована ринкова ціна смартфона: {round(predicted_price, 2)} INR")

        st.write("Введені характеристики:")
        st.dataframe(
            pd.DataFrame(list(new_phone.items()), columns=["Ознака", "Значення"]),
            use_container_width=True
        )


st.markdown("---")
st.write(
    "Програмний застосунок реалізує повний цикл аналізу: "
    "завантаження даних, очищення, формування ознак, статистику, графіки, "
    "кореляції, регресійну модель та прогнозування ціни."
)
