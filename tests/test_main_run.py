import importlib
import src.main


def test_main_runs_light(monkeypatch):
    """
    Проверяем, что main выполняется без ошибок,
    но без реального обучения.
    """
    monkeypatch.setattr(src.main.DataLoader, "load_data", lambda self: None)
    monkeypatch.setattr(src.main.DataLoader, "preprocess", lambda self: None)
    monkeypatch.setattr(
        src.main.DatabaseManager, "save_data", lambda self, df: None
    )
    monkeypatch.setattr(src.main.PriceModel, "train", lambda self, df: None)
    monkeypatch.setattr(src.main.PriceModel, "predict", lambda self, df: [12345])

    importlib.reload(src.main)
