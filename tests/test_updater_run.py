import src.updater as updater


def test_updater_runs(monkeypatch):
    """Проверяем, что updater запускается без ошибок."""
    called = []

    # Подменяем зависимости, чтобы не было реальных вычислений
    monkeypatch.setattr(
        updater,
        "DataLoader",
        lambda *a, **k: type(
            "X",
            (),
            {
                "load_data": lambda self: [],
                "preprocess": lambda self: [],
            },
        )(),
    )

    monkeypatch.setattr(
        updater,
        "DatabaseManager",
        lambda *a, **k: type(
            "X",
            (),
            {"save_data": lambda self, df: called.append("db")},
        )(),
    )

    monkeypatch.setattr(
        updater,
        "PriceModel",
        lambda *a, **k: type(
            "X",
            (),
            {"train": lambda self, df: called.append("model")},
        )(),
    )

    if hasattr(updater, "update_model"):
        updater.update_model()
    elif hasattr(updater, "main"):
        updater.main()
    else:
        raise AttributeError(
            "В src/updater.py нет функции update_model() или main()."
        )

    assert "model" in called or "db" in called
