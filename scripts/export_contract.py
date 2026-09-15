"""Export the public contract; --check is an offline CI drift check."""

import argparse
import json
from pathlib import Path

from app.core.config import Settings
from app.main import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1] / "contracts/openapi.json"
    text = (
        json.dumps(create_app(Settings(_env_file=None)).openapi(), indent=2, sort_keys=True) + "\n"
    )
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            raise SystemExit(
                "Public API contract changed; regenerate and review contracts/openapi.json"
            )
    else:
        target.parent.mkdir(exist_ok=True)
        target.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
