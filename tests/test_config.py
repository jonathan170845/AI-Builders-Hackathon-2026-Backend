from app.core.config import Settings


def test_frontend_origins_accept_comma_separated_environment_value(monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGINS", "http://localhost:5173, https://veritas.example/")

    settings = Settings()

    assert settings.frontend_origins == ["http://localhost:5173", "https://veritas.example"]


def test_host_and_port_are_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("APP_HOST", "0.0.0.0")
    monkeypatch.setenv("APP_PORT", "9000")

    settings = Settings()

    assert settings.app_host == "0.0.0.0"
    assert settings.app_port == 9000
