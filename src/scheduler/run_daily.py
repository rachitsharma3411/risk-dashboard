"""
scheduler/run_daily.py
----------------------
Lightweight scheduler that runs the derivatives risk pipeline once per day
at a configurable time (default: 16:30 ET, after US market close).

Usage
-----
Run directly::

    python scheduler/run_daily.py

Or with a custom run time::

    python scheduler/run_daily.py --time 16:30

The scheduler loops indefinitely, sleeping between checks, and delegates
the actual pipeline work to ``main.py`` via subprocess so each run gets a
clean Python environment.

To run as a background service on Linux::

    nohup python scheduler/run_daily.py &

Or via a system cron job (recommended for production)::

    30 16 * * 1-5  cd /path/to/repo && python main.py >> logs/risk.log 2>&1
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scheduler")


def _next_run_dt(run_time_str: str) -> datetime:
    """
    Compute the next datetime at which the pipeline should fire.

    If the target time for today has already passed, the next run is
    scheduled for tomorrow.

    Parameters
    ----------
    run_time_str : str
        Target time as "HH:MM" (24-hour clock, local timezone).

    Returns
    -------
    datetime
        The next run datetime (today or tomorrow depending on current time).
    """
    now = datetime.now()
    hh, mm = map(int, run_time_str.split(":"))
    candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def _run_pipeline() -> int:
    """
    Execute ``main.py`` as a subprocess and return its exit code.

    The script is assumed to be in the same directory as this scheduler file
    (i.e. one level up: ``../main.py`` relative to ``scheduler/``).

    Returns
    -------
    int
        The exit code of ``main.py`` (0 = success).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(script_dir, "..", "main.py")
    main_path = os.path.normpath(main_path)

    logger.info("Launching pipeline: %s", main_path)
    result = subprocess.run(
        [sys.executable, main_path],
        cwd=os.path.dirname(main_path),
    )
    return result.returncode


def run_scheduler(run_time_str: str = "16:30", run_once: bool = False) -> None:
    """
    Main scheduler loop.

    Sleeps until the target time, executes the pipeline, then waits for the
    next day's run.  Repeats indefinitely unless *run_once* is ``True``.

    Parameters
    ----------
    run_time_str : str, optional
        Daily run time as "HH:MM".  Default is "16:30".
    run_once : bool, optional
        If ``True``, run the pipeline immediately without waiting and exit.
        Useful for testing the scheduler integration without waiting a full day.
    """
    if run_once:
        logger.info("--run-once flag set: executing pipeline immediately.")
        rc = _run_pipeline()
        logger.info("Pipeline finished with exit code %d.", rc)
        return

    logger.info("Derivatives risk scheduler started. Daily run time: %s", run_time_str)

    while True:
        next_run = _next_run_dt(run_time_str)
        wait_seconds = (next_run - datetime.now()).total_seconds()
        logger.info(
            "Next pipeline run at %s (in %.0f seconds / %.2f hours).",
            next_run.strftime("%Y-%m-%d %H:%M:%S"),
            wait_seconds,
            wait_seconds / 3600,
        )

        # Sleep in 60-second intervals so we can log heartbeats and respond
        # to keyboard interrupts promptly.
        while datetime.now() < next_run:
            sleep_chunk = min(60, (next_run - datetime.now()).total_seconds())
            if sleep_chunk > 0:
                time.sleep(sleep_chunk)

        logger.info("=== Triggering daily risk pipeline ===")
        try:
            rc = _run_pipeline()
            if rc == 0:
                logger.info("Pipeline completed successfully.")
            else:
                logger.error("Pipeline exited with code %d.", rc)
        except Exception as exc:
            logger.error("Failed to launch pipeline: %s", exc)

        # Brief pause before computing the next run time to avoid re-firing
        # within the same minute.
        time.sleep(90)


def main() -> None:
    """
    CLI entry point.

    Parses ``--time`` and ``--run-once`` arguments, then starts the
    scheduler loop.
    """
    parser = argparse.ArgumentParser(
        description="Derivatives Risk Report — Daily Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--time",
        default="16:30",
        metavar="HH:MM",
        help="Daily run time in 24-hour local time (default: 16:30)",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run the pipeline immediately instead of waiting for the scheduled time",
    )
    args = parser.parse_args()
    run_scheduler(run_time_str=args.time, run_once=args.run_once)


if __name__ == "__main__":
    main()
