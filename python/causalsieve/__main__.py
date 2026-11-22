"""Allow invoking the CLI via ``python3 -m causalsieve``."""

from __future__ import annotations

from causalsieve.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
