"""Market clock constants shared by the live data, the strategy, and the screens."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), name="IST")
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)


def format_wall_clock(moment: datetime) -> str:
    """Label the computer time when the data was last updated."""
    return moment.strftime("%Y-%m-%d %H:%M:%S") + " local time on this computer"
