import os
import pandas as pd
from sqlalchemy import create_engine


class DatabaseManager:
    """
    Класс для управления SQLite базой данных.
    """

    def __init__(self, db_path: str = None):
        """
        :param db_path: путь к базе данных (по умолчанию ../data/pricing_data.db)
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(base_dir, "data", "pricing_data.db")

        # если путь передан — используем его (например, для тестов)
        self.db_path = db_path or default_path
        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def save_data(self, df, table_name="pricing_data"):
        """
        Сохраняет DataFrame в SQLite.
        """
        try:
            df.to_sql(
                table_name,
                self.engine,
                if_exists="replace",
                index=False,
                method="multi",
                chunksize=5000,
            )
            print(f"[INFO] Данные сохранены в таблицу '{table_name}'.")
        except Exception as e:
            print(f"[ERROR] Ошибка при сохранении данных: {e}")

    def load_data(self, table_name="pricing_data"):
        """
        Загружает данные из SQLite.
        """
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engine)
            print(f"[INFO] Данные загружены из таблицы '{table_name}'.")
            return df
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке данных: {e}")
            return pd.DataFrame()
