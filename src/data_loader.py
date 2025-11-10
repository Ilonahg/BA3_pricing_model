import pandas as pd


class DataLoader:
    """
    Класс для загрузки и предварительной обработки данных из CSV файла.
    """

    def __init__(self, file_path: str):
        """
        :param file_path: путь к CSV файлу с данными
        """
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Загружает CSV файл в DataFrame.
        :return: DataFrame с исходными данными
        """
        try:
            df = pd.read_csv(self.file_path)
            print(f"[INFO] Файл {self.file_path} успешно загружен.")
            print(f"[INFO] Размер данных: {df.shape}")

            # если нет колонки competitor_price — добавляем
            if "competitor_price" not in df.columns:
                print(
                    "[WARN] Колонка 'competitor_price' отсутствует или пуста — "
                    "создаём автоматически..."
                )
                # пример приближённой цены конкурента
                df["competitor_price"] = df["price"] * 0.95

            self.data = df
            return df

        except FileNotFoundError:
            print(f"[ERROR] Файл не найден: {self.file_path}")
            raise

        except Exception as e:
            print(f"[ERROR] Ошибка при чтении файла: {e}")
            raise

    def preprocess(self) -> pd.DataFrame:
        """
        Очищает и подготавливает данные к обучению модели.
        :return: обработанный DataFrame
        """
        if self.data is None:
            raise ValueError(
                "Сначала нужно загрузить данные методом load_data()"
            )

        df = self.data.copy()

        # Удаляем дубликаты
        df = df.drop_duplicates()

        # Удаляем пропуски в ключевых столбцах
        df = df.dropna(subset=["price", "count", "add_cost"])

        # Заполняем оставшиеся пропуски средними значениями
        df["price"] = df["price"].fillna(df["price"].mean())
        df["count"] = df["count"].fillna(df["count"].mean())
        df["add_cost"] = df["add_cost"].fillna(df["add_cost"].mean())
        df["competitor_price"] = df["competitor_price"].fillna(
            df["competitor_price"].mean()
        )

        # Кодируем текстовые признаки
        df["company"] = df["company"].astype("category").cat.codes
        df["product"] = df["product"].astype("category").cat.codes

        print(f"[INFO] Данные успешно обработаны. Размер: {df.shape}")
        self.data = df
        return df
