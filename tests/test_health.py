def test_live(client):
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready(client):
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_not_ready_returns_503_with_check_summary(client):
    client.app.state.readiness_checks = {"bootstrap": "not_ready"}

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"bootstrap": "not_ready"},
    }
