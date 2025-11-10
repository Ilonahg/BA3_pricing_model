import pytest
import pandas as pd
from src.data_loader import DataLoader


@pytest.fixture
def sample_csv(tmp_path):
    """Создаёт временный CSV для теста."""
    test_file = tmp_path / "test_data.csv"
    df = pd.DataFrame(
        {
            "price": [100, None, 300],
            "count": [10, 20, None],
            "add_cost": [5, None, 15],
            "company": ["A", "B", "C"],
            "product": ["X", "Y", "Z"],
        }
    )
    df.to_csv(test_file, index=False)
    return test_file


def test_load_data(sample_csv):
    """Проверяем загрузку данных."""
    loader = DataLoader(str(sample_csv))
    df = loader.load_data()
    assert not df.empty
    assert "price" in df.columns


def test_preprocess(sample_csv):
    """Проверяем очистку и подготовку данных."""
    loader = DataLoader(str(sample_csv))
    loader.load_data()
    df_clean = loader.preprocess()
    assert not df_clean.isna().any().any()
    assert "company" in df_clean.columns
