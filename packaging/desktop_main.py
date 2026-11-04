"""Entry point of the packaged Windows application."""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    multiprocessing.freeze_support()
    from nestor.desktop import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
