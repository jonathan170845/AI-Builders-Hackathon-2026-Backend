from __future__ import annotations

from uuid import UUID, uuid4

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


def payload_with_financial_input(**overrides):
    return {
        **VALID_PAYLOAD,
        "financialInputs": {**VALID_PAYLOAD["financialInputs"], **overrides},
    }


def test_create_analysis_returns_queued_job_and_location(client):
    response = client.post("/api/v1/analyses", json=VALID_PAYLOAD)

    assert response.status_code == 202
    body = response.json()
    UUID(body["id"])
    assert body["status"] == "queued"
    assert body["createdAt"].endswith("Z")
    assert response.headers["location"] == f"/api/v1/analyses/{body['id']}"


def test_created_analysis_can_be_polled_after_worker_failure(client):
    created = client.post("/api/v1/analyses", json=VALID_PAYLOAD).json()

    response = client.get(f"/api/v1/analyses/{created['id']}")

    assert response.status_code == 200
    assert response.status_code == 200
    assert response.json()["id"] == created["id"]
    assert response.json()["status"] == "failed"
    assert response.json()["error"]["code"] == "LLM_NOT_CONFIGURED"


def test_unknown_analysis_uses_error_envelope(client):
    response = client.get(f"/api/v1/analyses/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ANALYSIS_NOT_FOUND"
    assert response.json()["error"]["message"] == "Analysis was not found"
    UUID(response.json()["error"]["requestId"])


@pytest.mark.parametrize(
    "payload",
    [
        {**VALID_PAYLOAD, "decision": "   "},
        payload_with_financial_input(monthlyOrders=-1),
        payload_with_financial_input(monthlyOrders=1.5),
        payload_with_financial_input(monthlyOrders=10**12 + 1),
        payload_with_financial_input(monthlyOrders="NaN"),
        payload_with_financial_input(revenuePerOrder=10**15 + 1),
        payload_with_financial_input(revenuePerOrder=1.00001),
        {**VALID_PAYLOAD, "financialInputs": {"monthlyOrders": 1}},
        {**VALID_PAYLOAD, "unexpected": True},
    ],
)
def test_invalid_create_request_returns_validation_envelope(client, payload):
    response = client.post("/api/v1/analyses", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    UUID(response.json()["error"]["requestId"])


def test_oversized_request_uses_error_envelope(client):
    response = client.post(
        "/api/v1/analyses",
        content=b" " * (64 * 1024 + 1),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_invalid_analysis_id_uses_validation_error_envelope(client):
    response = client.get("/api/v1/analyses/not-a-uuid")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unknown_route_uses_error_envelope(client):
    response = client.get("/api/v1/not-a-route", headers={"X-Request-ID": "request-123"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
    assert response.json()["error"]["requestId"] == "request-123"
    assert response.headers["x-request-id"] == "request-123"


def test_method_not_allowed_uses_error_envelope(client):
    response = client.put("/api/v1/analyses")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "METHOD_NOT_ALLOWED"


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
    financial_inputs_schema = schema["components"]["schemas"]["FinancialInputs"]
    analysis_result_schema = schema["components"]["schemas"]["AnalysisResult"]
    failed_schema = schema["components"]["schemas"]["AnalysisFailed"]

    assert "financialInputs" in request_schema["properties"]
    assert "financial_inputs" not in request_schema["properties"]
    assert request_schema["properties"]["decision"]["minLength"] == 10
    assert financial_inputs_schema["properties"]["monthlyOrders"]["type"] == "integer"
    assert financial_inputs_schema["properties"]["monthlyOrders"]["maximum"] == 10**12
    assert analysis_result_schema["properties"]["financialResults"]["anyOf"][-1] == {"type": "null"}
    idx_benchmark_schema = schema["components"]["schemas"]["IdxBenchmark"]
    assert idx_benchmark_schema["properties"]["comparisonType"]["const"] == "directional"
    assert idx_benchmark_schema["properties"]["sampleSize"]["minimum"] == 1
    assert "status" not in analysis_result_schema["properties"]
    assert "error" in failed_schema["properties"]
    assert "stage" in failed_schema["required"]
