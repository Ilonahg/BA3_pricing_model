from src.data_loader import DataLoader
from src.database import DatabaseManager
from src.model import PriceModel


def update_model():
    """
    Обновляет модель на основе новых данных.
    Используется в тестах и может запускаться вручную.
    """
    print("[INFO] Запуск обновления модели...")

    # 1. Загружаем и подготавливаем данные
    loader = DataLoader("data/csv_data.csv")
    df = loader.load_data()
    df = loader.preprocess()

    # 2. Сохраняем в базу
    db = DatabaseManager()
    db.save_data(df)

    # 3. Переобучаем модель
    pm = PriceModel()
    pm.train(df)

    print("[INFO] Модель успешно обновлена ✅")


if __name__ == "__main__":
    update_model()
