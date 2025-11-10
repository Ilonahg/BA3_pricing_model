import time
from src.data_loader import DataLoader
from src.database import DatabaseManager
from src.model import PriceModel


def main():
    """Главный процесс ценообразования."""
    start_time = time.time()
    print("[INFO] --- Запуск основного процесса ценообразования ---")

    # 1. Загружаем и обрабатываем данные
    loader = DataLoader("data/csv_data.csv")
    df = loader.load_data()
    df = loader.preprocess()
    df = df.sample(frac=0.05, random_state=42)  # 5% данных для теста

    # 2. Сохраняем в базу
    db = DatabaseManager()
    db.save_data(df)

    # 3. Обучаем модель
    pm = PriceModel()
    pm.train(df)

    # 4. Пример прогноза
    example = df.drop(columns=["price"]).iloc[[0]]
    prediction = pm.predict(example)
    print(f"[INFO] Прогноз для первого примера: {prediction[0]:.2f}")

    elapsed = time.time() - start_time
    print("[INFO] --- Работа завершена успешно ✅ ---")
    print(f"[INFO] Время выполнения: {elapsed:.2f} сек.")


if __name__ == "__main__":
    main()
