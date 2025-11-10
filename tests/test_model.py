import pytest
from src.model import PriceModel


@pytest.fixture(scope="module")
def price_model():
    return PriceModel()


@pytest.fixture(scope="module")
def loaded_data(price_model):
    """Загружаем тестовую базу"""
    df = price_model.load_data()
    assert not df.empty, "Ошибка: данные из базы не загружены"
    return df


def test_data_loaded(loaded_data):
    """Проверяем, что данные загружены корректно"""
    assert "price" in loaded_data.columns
    assert len(loaded_data) > 0


def test_model_training(price_model, loaded_data):
    """Проверяем обучение и сохранение модели"""
    # берём все строки, если данных мало
    df_sample = loaded_data.copy()
    assert not df_sample.empty, "Данные для обучения пусты!"

    price_model.train(df_sample)
    assert price_model.model is not None


def test_model_prediction(price_model, loaded_data):
    """Проверяем работу предсказания"""
    X_new = loaded_data.drop(columns=["price"]).iloc[[0]]
    pred = price_model.predict(X_new)
    assert len(pred) == 1
    assert pred[0] > 0
