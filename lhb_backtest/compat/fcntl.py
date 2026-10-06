"""Small Windows compatibility shim for SimTradeData's process lock.

The upstream project imports POSIX-only :mod:`fcntl` unconditionally.  Its
only use is ``flock`` for a one-byte, non-blocking process lock, which maps to
``msvcrt.locking`` on Windows.
"""

from __future__ import annotations

import os

import msvcrt

LOCK_EX = 0x02
LOCK_NB = 0x04
LOCK_UN = 0x08


def flock(fd: int, operation: int) -> None:
    os.lseek(fd, 0, os.SEEK_SET)
    if operation & LOCK_UN:
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return

    # Windows cannot lock beyond EOF reliably. Ensure one lock byte exists.
    if os.fstat(fd).st_size == 0:
        os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
    mode = msvcrt.LK_NBLCK if operation & LOCK_NB else msvcrt.LK_LOCK
    try:
        msvcrt.locking(fd, mode, 1)
    except OSError as exc:
        raise BlockingIOError(str(exc)) from exc
    finally:
        os.lseek(fd, 0, os.SEEK_SET)
