"""Validate packaged artifacts and launch the production ASGI server."""

import os
from pathlib import Path

import uvicorn

from deployment.validate_bundle import validate_bundle


def main() -> None:
    bundle = Path(os.environ.get("CLAIMGUARD_BUNDLE", "/app"))
    validate_bundle(bundle)
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=int(os.environ.get("WEB_CONCURRENCY", "1")),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
