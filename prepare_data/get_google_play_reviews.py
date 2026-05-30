import time
import re
import pandas as pd
from tqdm import tqdm
from google_play_scraper import reviews, Sort


# Список приложений. ID лучше проверить вручную в URL Google Play.
APPS = [
    {"app_id": "org.telegram.messenger", "app_name": "Telegram"},
    {"app_id": "com.whatsapp", "app_name": "WhatsApp"},
    {"app_id": "com.vkontakte.android", "app_name": "VK"},
    {"app_id": "ru.yandex.taxi", "app_name": "Yandex Go"},
    {"app_id": "ru.yandex.yandexmaps", "app_name": "Yandex Maps"},
    {"app_id": "com.avito.android", "app_name": "Avito"},
]

MAX_REVIEWS_PER_APP = 500
LANG = "ru"
COUNTRY = "ru"


def looks_like_russian(text: str) -> bool:
    """
    Простая проверка, что текст похож на русский:
    считаем долю кириллических символов среди букв.
    """
    if not isinstance(text, str):
        return False

    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    if not letters:
        return False

    cyrillic = re.findall(r"[А-Яа-яЁё]", text)
    return len(cyrillic) / len(letters) >= 0.5


def collect_reviews_for_app(app_id: str, app_name: str, max_reviews: int = 500) -> list[dict]:
    """
    Собирает отзывы для одного приложения из Google Play.
    Возвращает список словарей.
    """
    collected = []
    continuation_token = None

    while len(collected) < max_reviews:
        count = min(200, max_reviews - len(collected))

        try:
            batch, continuation_token = reviews(
                app_id,
                lang=LANG,
                country=COUNTRY,
                sort=Sort.NEWEST,
                count=count,
                continuation_token=continuation_token,
            )
        except Exception as e:
            print(f"[ERROR] {app_name} ({app_id}): {e}")
            break

        if not batch:
            break

        for item in batch:
            text = item.get("content", "")

            # Убираем совсем пустые отзывы
            if not text or len(text.strip()) < 2:
                continue

            # Фильтр на русский язык
            if not looks_like_russian(text):
                continue

            collected.append({
                "source": "google_play",
                "app_id": app_id,
                "app_name": app_name,
                "review_id": item.get("reviewId"),
                "text": text.strip(),
                "rating": item.get("score"),
                "thumbs_up_count": item.get("thumbsUpCount"),
                "review_created_version": item.get("reviewCreatedVersion"),
                "app_version": item.get("appVersion"),
                "date": item.get("at"),
                "reply_content": item.get("replyContent"),
                "replied_at": item.get("repliedAt"),

                # Эти поля потом заполнишь вручную или полуавтоматически
                "label_gold": "",
                "summary_gold": "",
            })

        print(f"{app_name}: собрано {len(collected)} отзывов")

        if continuation_token is None:
            break

        # Небольшая пауза, чтобы не спамить запросами
        time.sleep(1)

    return collected


def main():
    all_reviews = []

    for app in tqdm(APPS):
        app_reviews = collect_reviews_for_app(
            app_id=app["app_id"],
            app_name=app["app_name"],
            max_reviews=MAX_REVIEWS_PER_APP,
        )
        all_reviews.extend(app_reviews)

    df = pd.DataFrame(all_reviews)

    if df.empty:
        print("Отзывы не собраны. Проверь app_id, страну, язык или доступность приложений.")
        return

    # Удаляем дубликаты по тексту внутри одного приложения
    df["text_normalized"] = (
        df["text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = df.drop_duplicates(subset=["app_id", "text_normalized"])
    df = df.drop(columns=["text_normalized"])

    # Перемешиваем датасет
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Сохраняем в CSV и Excel
    df.to_csv("data/app_reviews_ru_raw.csv", index=False, encoding="utf-8-sig")
    df.to_excel("data/app_reviews_ru_raw.xlsx", index=False)

    print(f"Итоговый размер датасета: {len(df)} отзывов")
    print("Файлы сохранены: app_reviews_ru_raw.csv и app_reviews_ru_raw.xlsx")


if __name__ == "__main__":
    main()