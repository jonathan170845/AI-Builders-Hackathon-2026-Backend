"""Opt-in synthetic API/provider smoke test; prints status/counts, never credentials."""

import argparse
import json
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument(
    "--live", action="store_true", help="Create a synthetic job (calls configured provider)"
)
parser.add_argument("--financial", action="store_true")
parser.add_argument("--base-url", default="http://127.0.0.1:8000")
args = parser.parse_args()
base = args.base_url.rstrip("/")


def get(path):
    with urllib.request.urlopen(base + path, timeout=30) as response:
        return json.load(response)


print("readiness", get("/health/ready"), flush=True)
if not args.live:
    print("history_count", len(get("/api/v1/analyses")["items"]))
    raise SystemExit(0)
payload = {
    "decision": (
        "Technical smoke test: pilot a small business scheduling software subscription "
        "with ten local shops before expanding to a second city."
    )
}
if args.financial:
    payload["financialInputs"] = dict(
        monthlyOrders=100,
        revenuePerOrder=10,
        variableCostPerOrder=3,
        promoSubsidy=0,
        deliveryCost=1,
        fixedCost=400,
        driverCost=100,
        cashBalance=5000,
    )
request = urllib.request.Request(
    base + "/api/v1/analyses",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
started = time.perf_counter()
with urllib.request.urlopen(request, timeout=30) as response:
    accepted = json.load(response)
    print(
        "create",
        response.status,
        "seconds",
        round(time.perf_counter() - started, 3),
        "id",
        accepted["id"],
        flush=True,
    )
last = None
while time.perf_counter() - started < 620:
    detail = get("/api/v1/analyses/" + accepted["id"])
    current = (detail["status"], detail["stage"])
    if current != last:
        print("state", current, "elapsed", round(time.perf_counter() - started, 1), flush=True)
        last = current
    if detail["status"] == "failed":
        print("error", detail["error"], flush=True)
        raise SystemExit(1)
    if detail["status"] == "completed":
        result = detail["result"]
        assert result["id"] == accepted["id"]
        if args.financial:
            assert result["financialResults"]["operatingProfit"] == 100
            assert result["financialResults"]["breakEvenOrders"] == 84
        else:
            assert result["financialResults"] is None
        assert any(item["id"] == accepted["id"] for item in get("/api/v1/analyses")["items"])
        print(
            "completed",
            {
                "assumptions": result["assumptionCount"],
                "sources": len(result["sources"]),
                "financialResults": result["financialResults"],
            },
            flush=True,
        )
        break
    time.sleep(2)
else:
    raise SystemExit("Job still pending; keep the printed ID for later polling")
