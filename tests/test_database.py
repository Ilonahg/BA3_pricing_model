import pytest
import pandas as pd
from src.database import DatabaseManager


@pytest.fixture
def sample_df(tmp_path):
    """Создаёт примерный DataFrame для тестов."""
    data = {
        "price": [100, 200, 300],
        "count": [10, 20, 30],
        "add_cost": [5, 10, 15],
        "company": [1, 1, 2],
        "product": [2, 3, 4],
        "competitor_price": [95, 190, 280],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


def test_save_and_load_data(tmp_path, sample_df):
    """Проверяем сохранение и загрузку данных из временной БД."""
    _, df = sample_df
    db_path = tmp_path / "test_pricing_data.db"
    db = DatabaseManager(db_path=str(db_path))
    db.save_data(df)

    df_loaded = db.load_data()
    assert not df_loaded.empty
    assert df_loaded.shape == df.shape


def test_database_file_created(tmp_path, sample_df):
    """Проверяем, что база данных физически создаётся."""
    _, df = sample_df
    db_path = tmp_path / "check_db.db"

    db = DatabaseManager(db_path=str(db_path))
    db.save_data(df)

    assert db_path.exists(), "Файл базы данных не был создан!"
