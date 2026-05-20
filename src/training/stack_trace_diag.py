"""Register faulthandler on SIGUSR1 so live train/worker PIDs can dump Python stacks.

Usage (from another terminal)::

    kill -USR1 $(cat runs/realbg/dist_ppo_014/pid)      # host
    kill -USR1 <worker_pid>                             # one worker

Stacks go to stderr (the train terminal). Install py-spy for richer dumps without signals::

    pipx install py-spy
    sudo py-spy dump --pid $(cat runs/realbg/dist_ppo_014/pid)
"""

from __future__ import annotations

import faulthandler
import signal
import sys
import threading


def enable_stack_dump_on_signal() -> None:
    """Enable traceback dumps on SIGUSR1 (all threads when supported)."""
    faulthandler.enable(file=sys.stderr, all_threads=True)
    try:
        faulthandler.register(
            signal.SIGUSR1,
            file=sys.stderr,
            all_threads=True,
            chain=True,
        )
    except AttributeError:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, chain=True)

    # Periodic watchdog is opt-in via env (see scripts/diagnose_hung_train.sh).
    if _watchdog_enabled():
        t = threading.Thread(target=_stack_watchdog_loop, name="stack-watchdog", daemon=True)
        t.start()


def _watchdog_enabled() -> bool:
    import os

    return os.environ.get("RL_STACK_WATCHDOG_SEC", "").strip().isdigit()


def _stack_watchdog_loop() -> None:
    import os
    import time

    interval = max(30, int(os.environ["RL_STACK_WATCHDOG_SEC"]))
    while True:
        time.sleep(interval)
        print(
            f"\n=== RL_STACK_WATCHDOG ({interval}s) thread dump ===",
            file=sys.stderr,
            flush=True,
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
