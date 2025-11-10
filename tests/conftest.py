import pytest
import pandas as pd
from src.database import DatabaseManager
from sqlalchemy import create_engine
import os


@pytest.fixture(scope="session", autouse=True)
def setup_test_database(tmp_path_factory):
    """
    Создаёт временную базу данных с тестовыми данными.
    Подменяет путь, чтобы тесты не использовали реальную pricing_data.db
    """
    tmp_dir = tmp_path_factory.mktemp("db")
    test_db = tmp_dir / "test_pricing_data.db"

    df = pd.DataFrame(
        {
            "price": [100, 200, 300],
            "count": [10, 20, 30],
            "add_cost": [5, 10, 15],
            "company": [1, 2, 3],
            "product": [11, 12, 13],
            "competitor_price": [90, 180, 250],
        }
    )

    db = DatabaseManager()
    db.db_path = str(test_db)
    db.engine = create_engine(f"sqlite:///{db.db_path}")
    db.save_data(df)

    os.environ["TEST_DB_PATH"] = str(test_db)
    return str(test_db)
