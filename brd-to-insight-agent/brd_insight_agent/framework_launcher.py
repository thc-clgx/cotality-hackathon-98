"""Start the BRD-to-BigQuery framework with its progress logging switched on.

The framework's ``managed_runner`` entry point never calls its own ``setup_logging``,
so Python's root logger stays at WARNING and every ``logger.info`` call it makes is
discarded. Only the final JSON summary reaches standard output, which makes a run that
takes minutes look like it is doing nothing.

This launcher configures logging to stdout *before* handing control to the framework,
so the adapter can show progress while the run is happening. It changes nothing else:
arguments, exit codes and the JSON summary pass straight through.

Usage, as the ``BRD_AGENT_CMD`` for ``cli`` mode::

    <framework venv>/bin/python <this file> -m src.orchestrator.managed_runner \
        --brd {brd} --source-root {source_dir}

Set ``BRD_AGENT_LOG_LEVEL`` to change the level (default ``INFO``).
"""

from __future__ import annotations

import logging
import os
import runpy
import sys

# Chatty at INFO and nothing to do with pipeline progress; the framework's own
# setup_logging quietens exactly these.
NOISY = ("google.auth", "google.cloud", "google_genai", "urllib3", "httpx",
         "charset_normalizer", "grpc", "fsspec")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2 or arguments[0] != "-m":
        print("Usage: framework_launcher.py -m <module> [args...]", file=sys.stderr)
        return 2
    module = arguments[1]

    # The framework is imported from its own working directory, exactly as 'python -m'
    # would, which running a script file does not do on its own.
    sys.path.insert(0, os.getcwd())
    level = (os.environ.get("BRD_AGENT_LOG_LEVEL") or "INFO").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        level = "INFO"
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)-7s | %(name)-34s | %(message)s",
        force=True,
    )
    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    # argv[0] becomes the module, so the framework's own argument parsing is unchanged.
    sys.argv = [module] + arguments[2:]
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
