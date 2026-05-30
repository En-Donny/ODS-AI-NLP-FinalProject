import re
import html
import json
import argparse
import unicodedata
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd


@dataclass
class CleaningConfig:
    # Минимальная длина текста, который точно можно отдавать на разметку
    min_text_len: int = 10

    # Минимальная длина совсем коротких отзывов.
    # Всё, что короче, будет удалено.
    absolute_min_text_len: int = 3

    # Порог доли кириллицы среди букв.
    # 0.45 означает, что минимум 45% букв должны быть кириллическими.
    min_cyrillic_ratio: float = 0.45

    # Максимальная доля латиницы.
    # Полезно, чтобы отсекать английские отзывы.
    max_latin_ratio: float = 0.50

    # Минимальное количество букв/цифр в отзыве.
    # Отзывы из одних смайлов/точек/знаков будут отделены.
    min_alnum_count: int = 2

    # Удалять дубликаты внутри одного приложения.
    # Это безопаснее, чем удалять одинаковые тексты глобально.
    deduplicate_within_app: bool = True

    # Сколько коротких отзывов оставить как кандидаты для класса rating.
    max_short_rating_candidates: int = 300

    # Случайное зерно для воспроизводимости
    random_state: int = 42


CONFIG = CleaningConfig()


TEXT_COLUMN_CANDIDATES = [
    "text",
    "review",
    "review_text",
    "content",
    "comment",
    "отзыв",
    "текст",
]

APP_ID_CANDIDATES = [
    "app_id",
    "application_id",
    "package",
    "package_name",
]

APP_NAME_CANDIDATES = [
    "app_name",
    "application_name",
    "name",
    "приложение",
]

RATING_CANDIDATES = [
    "rating",
    "score",
    "stars",
    "оценка",
]


def read_dataset(input_path: str) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError("Поддерживаются только .csv, .xlsx и .xls файлы")


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    columns_lower = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]

    if required:
        raise ValueError(
            f"Не найдена обязательная колонка. Возможные названия: {candidates}. "
            f"Колонки в файле: {list(df.columns)}"
        )

    return None


def normalize_unicode(text: str) -> str:
    """
    Нормализует Unicode:
    - приводит разные формы символов к единому виду;
    - полезно для странных кавычек, пробелов и символов.
    """
    return unicodedata.normalize("NFKC", text)


def remove_control_chars(text: str) -> str:
    """
    Удаляет управляющие и невидимые символы.
    """
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")
    text = text.replace("\xa0", " ")

    cleaned_chars = []

    for char in text:
        category = unicodedata.category(char)

        # Cc — управляющие символы, Cf — форматирующие невидимые символы
        if category in ["Cc", "Cf"]:
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(char)

    return "".join(cleaned_chars)


def remove_html(text: str) -> str:
    """
    Удаляет HTML-теги и декодирует HTML-сущности.
    """
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def mask_personal_data(text: str) -> str:
    """
    Маскирует потенциальные персональные данные:
    - email;
    - URL;
    - телефонные номера.
    """

    # Email
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "<EMAIL>",
        text,
    )

    # URL
    text = re.sub(
        r"(https?://\S+|www\.\S+)",
        "<URL>",
        text,
        flags=re.IGNORECASE,
    )

    # Telegram/соцсети вида @username
    text = re.sub(
        r"(?<!\w)@[A-Za-zА-Яа-яЁё0-9_]{4,32}",
        "<USERNAME>",
        text,
    )

    # Российские номера телефонов и похожие длинные номера
    phone_pattern = r"""
        (?:
            (?:\+7|7|8)
            [\s\-()]*
        )?
        (?:
            \d[\s\-()]*
        ){10}
    """

    text = re.sub(
        phone_pattern,
        "<PHONE>",
        text,
        flags=re.VERBOSE,
    )

    return text


def reduce_repeated_chars(text: str) -> str:
    """
    Сжимает чрезмерные повторы, но не убивает эмоциональность полностью.

    Пример:
    - "крууууутооооо" -> "круутоо"
    - "!!!!!!" -> "!!!"
    """

    # Повторы букв/цифр: больше 3 подряд сжимаем до 2
    text = re.sub(r"([A-Za-zА-Яа-яЁё0-9])\1{3,}", r"\1\1", text)

    # Повторы знаков препинания: больше 3 подряд сжимаем до 3
    text = re.sub(r"([!?.,])\1{3,}", r"\1\1\1", text)

    return text


def normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """
    Основная мягкая очистка.
    Важно: не удаляем стоп-слова, не лемматизируем, не приводим к lowercase.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = remove_html(text)
    text = mask_personal_data(text)
    text = reduce_repeated_chars(text)
    text = normalize_spaces(text)

    return text


def count_cyrillic(text: str) -> int:
    return len(re.findall(r"[А-Яа-яЁё]", text))


def count_latin(text: str) -> int:
    return len(re.findall(r"[A-Za-z]", text))


def count_letters(text: str) -> int:
    return len(re.findall(r"[A-Za-zА-Яа-яЁё]", text))


def count_alnum(text: str) -> int:
    return len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]", text))


def cyrillic_ratio(text: str) -> float:
    letters = count_letters(text)

    if letters == 0:
        return 0.0

    return count_cyrillic(text) / letters


def latin_ratio(text: str) -> float:
    letters = count_letters(text)

    if letters == 0:
        return 0.0

    return count_latin(text) / letters


def is_emoji_or_symbols_only(text: str) -> bool:
    """
    Проверяет, состоит ли отзыв почти полностью из эмодзи/символов/пунктуации.
    """
    if not text:
        return False

    return count_alnum(text) == 0


def is_low_information_junk(text: str) -> bool:
    """
    Отсекает явный мусор, но НЕ удаляет длинные информативные отзывы.

    Удаляем:
    - пустые строки;
    - строки только из эмодзи/пунктуации;
    - строки из одного повторяющегося символа;
    - короткие строки с очень низким разнообразием символов.

    Не удаляем:
    - длинные отзывы;
    - отзывы с нормальными словами;
    - короткие осмысленные оценки вроде "ужасно", "отлично", "не работает".
    """

    if not text:
        return True

    text = str(text).strip()

    if not text:
        return True

    normalized = re.sub(r"\s+", "", text.lower())

    if not normalized:
        return True

    # Только символы, эмодзи, пунктуация без букв и цифр
    if is_emoji_or_symbols_only(text):
        return True

    alnum_chars = re.findall(r"[A-Za-zА-Яа-яЁё0-9]", normalized)

    if len(alnum_chars) == 0:
        return True

    # Если после очистки строка состоит из одного символа,
    # например "аааааааа", ".....", "111111"
    if len(set(alnum_chars)) == 1 and len(alnum_chars) >= 4:
        return True

    # Достаём слова
    words = re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", text.lower())
    cyrillic_words = re.findall(r"[А-Яа-яЁё]{2,}", text.lower())

    # Если есть несколько нормальных слов, это уже не мусор.
    # Например:
    # "не работает"
    # "приложение сломано"
    # "после обновления не открывается"
    if len(words) >= 2:
        return False

    # Одно короткое, но осмысленное русское слово тоже лучше не удалять.
    # Например:
    # "ужасно", "отлично", "нормально", "плохо"
    if len(cyrillic_words) == 1:
        word = cyrillic_words[0]

        if len(word) >= 4:
            return False

    # Проверку на низкое разнообразие символов применяем ТОЛЬКО к коротким строкам.
    # Для длинных отзывов это правило применять нельзя.
    if len(alnum_chars) <= 25:
        unique_ratio = len(set(alnum_chars)) / len(alnum_chars)

        if unique_ratio <= 0.25:
            return True

    # Дополнительный фильтр для короткого бессмысленного текста без русских букв
    # Например: "jdjdjdjd", "qwerty", "ahaha"
    cyr_count = count_cyrillic(text)
    lat_count = count_latin(text)

    if len(alnum_chars) <= 20 and cyr_count == 0 and lat_count > 0:
        # Если это короткая латиница без кириллицы, для русского датасета считаем мусором
        return True

    return False


def make_dedup_key(text: str) -> str:
    """
    Создаёт ключ для удаления почти одинаковых отзывов.

    Пример:
    - "Не работает!!!"
    - "не работает"
    - "Не   работает..."
    будут иметь близкий ключ.
    """
    text = text.lower()
    text = text.replace("ё", "е")

    # Убираем маски, чтобы одинаковые отзывы с разными телефонами/ссылками схлопывались
    text = text.replace("<email>", " ")
    text = text.replace("<phone>", " ")
    text = text.replace("<url>", " ")
    text = text.replace("<username>", " ")

    text = re.sub(r"[^a-zа-я0-9]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def classify_cleaning_status(text: str, config: CleaningConfig) -> tuple[str, str]:
    """
    Возвращает:
    - статус;
    - причину.

    Статусы:
    - keep
    - short_rating_candidate
    - drop
    """

    if not text:
        return "drop", "empty_after_cleaning"

    text_len = len(text)

    if text_len < config.absolute_min_text_len:
        return "drop", "too_short_absolute"

    if is_low_information_junk(text):
        return "drop", "low_information_junk"

    if count_alnum(text) < config.min_alnum_count:
        return "drop", "not_enough_alnum"

    cyr_ratio = cyrillic_ratio(text)
    lat_ratio = latin_ratio(text)

    # Если букв нет, это, скорее всего, неинформативный отзыв
    if count_letters(text) == 0:
        return "drop", "no_letters"

    # Отсеиваем явно нерусские отзывы
    if cyr_ratio < config.min_cyrillic_ratio and lat_ratio > config.max_latin_ratio:
        return "drop", "probably_not_russian"

    # Короткие, но осмысленные отзывы можно оставить отдельно как rating-кандидаты
    if text_len < config.min_text_len:
        return "short_rating_candidate", "short_but_possible_rating"

    return "keep", "ok"


def prepare_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет колонки для ручной разметки.
    """
    if "label_gold" not in df.columns:
        df["label_gold"] = ""

    if "summary_gold" not in df.columns:
        df["summary_gold"] = ""

    if "annotation_comment" not in df.columns:
        df["annotation_comment"] = ""

    if "is_checked_by_human" not in df.columns:
        df["is_checked_by_human"] = ""

    return df


def clean_reviews_dataset(input_path: str, output_dir: str, config: CleaningConfig) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = read_dataset(input_path)

    text_col = find_column(df, TEXT_COLUMN_CANDIDATES, required=True)
    app_id_col = find_column(df, APP_ID_CANDIDATES, required=False)
    app_name_col = find_column(df, APP_NAME_CANDIDATES, required=False)
    rating_col = find_column(df, RATING_CANDIDATES, required=False)

    original_rows_count = len(df)

    # Сохраняем исходный текст
    df["review_text_raw"] = df[text_col].astype(str)

    # Создаём очищенный текст
    df["review_text_clean"] = df["review_text_raw"].apply(clean_text)

    # Технические признаки качества
    df["text_len"] = df["review_text_clean"].str.len()
    df["cyrillic_ratio"] = df["review_text_clean"].apply(cyrillic_ratio)
    df["latin_ratio"] = df["review_text_clean"].apply(latin_ratio)
    df["alnum_count"] = df["review_text_clean"].apply(count_alnum)
    df["dedup_key"] = df["review_text_clean"].apply(make_dedup_key)

    # Статус очистки
    statuses = df["review_text_clean"].apply(lambda x: classify_cleaning_status(x, config))
    df["cleaning_status"] = statuses.apply(lambda x: x[0])
    df["cleaning_reason"] = statuses.apply(lambda x: x[1])

    # Разделяем по статусам
    keep_df = df[df["cleaning_status"] == "keep"].copy()
    short_df = df[df["cleaning_status"] == "short_rating_candidate"].copy()
    dropped_df = df[df["cleaning_status"] == "drop"].copy()

    # Удаление дубликатов
    before_dedup = len(keep_df)

    if app_id_col and config.deduplicate_within_app:
        dedup_subset = [app_id_col, "dedup_key"]
    elif app_name_col and config.deduplicate_within_app:
        dedup_subset = [app_name_col, "dedup_key"]
    else:
        dedup_subset = ["dedup_key"]

    duplicated_mask = keep_df.duplicated(subset=dedup_subset, keep="first")

    duplicates_df = keep_df[duplicated_mask].copy()
    duplicates_df["cleaning_status"] = "drop"
    duplicates_df["cleaning_reason"] = "duplicate"

    keep_df = keep_df[~duplicated_mask].copy()

    after_dedup = len(keep_df)

    # Короткие отзывы: оставим ограниченное количество как rating-кандидатов
    if len(short_df) > 0:
        short_df = short_df.sample(
            n=min(config.max_short_rating_candidates, len(short_df)),
            random_state=config.random_state,
        ).copy()

        short_df["label_gold"] = ""
        short_df["summary_gold"] = ""
        short_df["annotation_comment"] = "Короткий отзыв. Вероятный кандидат для rating."
        short_df["is_checked_by_human"] = ""

    # Основной датасет для разметки
    keep_df = prepare_annotation_columns(keep_df)

    # # Перемешиваем основной датасет
    # keep_df = keep_df.sample(frac=1, random_state=config.random_state).reset_index(drop=True)

    # Удалённые отзывы
    removed_df = pd.concat([dropped_df, duplicates_df], ignore_index=True)

    # Удобный порядок колонок
    preferred_columns = []

    for col in [
        app_name_col,
        app_id_col,
        rating_col,
        "review_text_raw",
        "review_text_clean",
        "label_gold",
        "summary_gold",
        "annotation_comment",
        "is_checked_by_human",
        "cleaning_status",
        "cleaning_reason",
        "text_len",
        "cyrillic_ratio",
        "latin_ratio",
        "alnum_count",
    ]:
        if col and col in keep_df.columns and col not in preferred_columns:
            preferred_columns.append(col)

    other_columns = [col for col in keep_df.columns if col not in preferred_columns and col != "dedup_key"]

    keep_df = keep_df[preferred_columns + other_columns]

    if len(short_df) > 0:
        short_preferred = [col for col in preferred_columns if col in short_df.columns]
        short_other = [col for col in short_df.columns if col not in short_preferred and col != "dedup_key"]
        short_df = short_df[short_preferred + short_other]

    if len(removed_df) > 0:
        removed_df = removed_df.drop(columns=["dedup_key"], errors="ignore")

    # Сохраняем результаты
    keep_csv = output_path / "reviews_for_gold_annotation.csv"
    keep_xlsx = output_path / "reviews_for_gold_annotation.xlsx"

    short_csv = output_path / "short_rating_candidates.csv"
    short_xlsx = output_path / "short_rating_candidates.xlsx"

    removed_csv = output_path / "removed_reviews.csv"
    removed_xlsx = output_path / "removed_reviews.xlsx"

    stats_json = output_path / "cleaning_stats.json"

    keep_df.to_csv(keep_csv, index=False, encoding="utf-8-sig")
    keep_df.to_excel(keep_xlsx, index=False)

    if len(short_df) > 0:
        short_df.to_csv(short_csv, index=False, encoding="utf-8-sig")
        short_df.to_excel(short_xlsx, index=False)

    if len(removed_df) > 0:
        removed_df.to_csv(removed_csv, index=False, encoding="utf-8-sig")
        removed_df.to_excel(removed_xlsx, index=False)

    stats = {
        "input_file": str(input_path),
        "output_dir": str(output_path),
        "config": asdict(config),
        "columns_detected": {
            "text_column": text_col,
            "app_id_column": app_id_col,
            "app_name_column": app_name_col,
            "rating_column": rating_col,
        },
        "rows": {
            "original": original_rows_count,
            "kept_for_annotation": len(keep_df),
            "short_rating_candidates": len(short_df),
            "removed_total": len(removed_df),
            "duplicates_removed": before_dedup - after_dedup,
        },
        "removed_reasons": (
            removed_df["cleaning_reason"].value_counts().to_dict()
            if len(removed_df) > 0
            else {}
        ),
    }

    with open(stats_json, "w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=4)

    print("Очистка завершена.")
    print(f"Исходных строк: {original_rows_count}")
    print(f"Оставлено для gold-разметки: {len(keep_df)}")
    print(f"Коротких rating-кандидатов: {len(short_df)}")
    print(f"Удалено всего: {len(removed_df)}")
    print(f"Удалено дубликатов: {before_dedup - after_dedup}")
    print()
    print(f"Основной файл CSV:  {keep_csv}")
    print(f"Основной файл Excel: {keep_xlsx}")
    print(f"Статистика: {stats_json}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Очистка русскоязычных отзывов приложений перед ручной gold-разметкой."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Путь к исходному CSV/XLSX файлу с отзывами.",
    )

    parser.add_argument(
        "--output-dir",
        default="cleaned_reviews",
        help="Папка для сохранения очищенных файлов.",
    )

    parser.add_argument(
        "--min-text-len",
        type=int,
        default=CONFIG.min_text_len,
        help="Минимальная длина текста для основного датасета разметки.",
    )

    parser.add_argument(
        "--absolute-min-text-len",
        type=int,
        default=CONFIG.absolute_min_text_len,
        help="Минимальная длина текста, ниже которой отзыв удаляется.",
    )

    parser.add_argument(
        "--min-cyrillic-ratio",
        type=float,
        default=CONFIG.min_cyrillic_ratio,
        help="Минимальная доля кириллицы среди букв.",
    )

    parser.add_argument(
        "--max-short-rating-candidates",
        type=int,
        default=CONFIG.max_short_rating_candidates,
        help="Сколько коротких отзывов сохранить отдельно как rating-кандидаты.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = CleaningConfig(
        min_text_len=args.min_text_len,
        absolute_min_text_len=args.absolute_min_text_len,
        min_cyrillic_ratio=args.min_cyrillic_ratio,
        max_short_rating_candidates=args.max_short_rating_candidates,
    )

    clean_reviews_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()