import os
import joblib
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from src.database import DatabaseManager


class PriceModel:
    """
    Класс для обучения и предсказания ценовой модели.
    """

    def __init__(self):
        """Инициализация модели и подключение к базе данных."""
        db_path = os.getenv("TEST_DB_PATH")
        self.db = DatabaseManager(db_path=db_path)
        self.model = None

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, "models", "pricing_model.pkl")

    def load_data(self):
        """Загружает данные из базы."""
        df = self.db.load_data()
        print(f"[INFO] Данные загружены из базы. Размер: {df.shape}")
        return df

    def train(self, df: pd.DataFrame):
        """Обучает модель RandomForestRegressor и сохраняет её."""
        if df.empty:
            raise ValueError("DataFrame пуст, невозможно обучить модель!")

        X = df.drop(columns=["price"])
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("[INFO] Модель обучена ✅")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

        # Сохраняем обученную модель
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"[INFO] Модель сохранена в: {self.model_path}")

    def predict(self, X_new: pd.DataFrame):
        """Делает предсказание на основе обученной модели."""
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError("Модель не обучена и не найдена!")
            self.model = joblib.load(self.model_path)

        preds = self.model.predict(X_new)
        return preds
