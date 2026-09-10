from __future__ import annotations

from uuid import UUID

import pytest


VALID_PAYLOAD = {
    "decision": "Scale on-demand grocery delivery to five new cities",
    "financialInputs": {
        "monthlyOrders": 1_000_000,
        "revenuePerOrder": 50,
        "variableCostPerOrder": 35,
        "promoSubsidy": 10,
        "deliveryCost": 8,
        "fixedCost": 5_000_000,
        "driverCost": 3_000_000,
        "cashBalance": 500_000_000,
    },
}


def test_create_analysis_returns_queued_job_and_location(client):
    response = client.post("/api/v1/analyses", json=VALID_PAYLOAD)

    assert response.status_code == 202
    body = response.json()
    UUID(body["id"])
    assert body["status"] == "queued"
    assert body["createdAt"].endswith("Z")
    assert response.headers["location"] == f"/api/v1/analyses/{body['id']}"


def test_created_analysis_can_be_polled(client):
    created = client.post("/api/v1/analyses", json=VALID_PAYLOAD).json()

    response = client.get(f"/api/v1/analyses/{created['id']}")

    assert response.status_code == 200
    assert response.json() == {
        "id": created["id"],
        "status": "queued",
        "stage": "queued",
        "createdAt": created["createdAt"],
        "updatedAt": created["createdAt"],
    }


def test_unknown_analysis_uses_error_envelope(client):
    response = client.get("/api/v1/analyses/missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ANALYSIS_NOT_FOUND"
    assert response.json()["error"]["message"] == "Analysis was not found"
    UUID(response.json()["error"]["requestId"])


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({**VALID_PAYLOAD, "decision": "   "}, "decision"),
        ({**VALID_PAYLOAD, "financialInputs": {**VALID_PAYLOAD["financialInputs"], "monthlyOrders": -1}}, "monthlyOrders"),
        ({**VALID_PAYLOAD, "financialInputs": {**VALID_PAYLOAD["financialInputs"], "monthlyOrders": "NaN"}}, "monthlyOrders"),
        ({**VALID_PAYLOAD, "unexpected": True}, "unexpected"),
    ],
)
def test_invalid_create_request_returns_validation_envelope(client, payload, field):
    response = client.post("/api/v1/analyses", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert field not in response.json()["error"]


def test_cors_allows_configured_origin(client):
    response = client.options(
        "/api/v1/analyses",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client):
    response = client.options(
        "/api/v1/analyses",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_openapi_preserves_camel_case_contract(client):
    schema = client.get("/openapi.json").json()
    request_schema = schema["components"]["schemas"]["CreateAnalysisRequest"]
    result_schema = schema["components"]["schemas"]["FinancialResults"]

    assert "financialInputs" in request_schema["properties"]
    assert "financial_inputs" not in request_schema["properties"]
    assert result_schema["properties"]["runwayMonths"]["anyOf"][-1] == {"type": "null"}
