from fastapi.testclient import TestClient

from app import main
from app.core.config import Settings


def test_unexpected_exception_uses_generic_error_envelope():
    app = main.create_app(Settings(frontend_origins=["http://localhost:5173"]))

    @app.get("/api/v1/test-unexpected-error")
    async def raise_unexpected_error() -> None:
        raise RuntimeError("internal details must not reach the client")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/test-unexpected-error")

    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "INTERNAL_SERVER_ERROR",
        "message": "An unexpected error occurred",
        "requestId": response.headers["x-request-id"],
    }


def test_run_uses_host_port_and_environment_settings(monkeypatch):
    captured = {}
    settings = Settings(app_env="production", app_host="0.0.0.0", app_port=9000)

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main.uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))

    main.run()

    assert captured == {"host": "0.0.0.0", "port": 9000, "reload": False}
