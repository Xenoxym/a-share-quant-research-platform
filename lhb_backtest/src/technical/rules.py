"""Dated, isolated exit model for a main-board holding that becomes ST.

These are rule-derived prices, never exchange-observed price limits. This does
not promote quarantined source prices in the existing clean dataset.
"""
from __future__ import annotations

import math
import re

RULE_SOURCES = [
    "https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/c_20230421_5720459.shtml",
    "https://www.szse.cn/lawrules/rule/repeal/rules/P020231230545143336292.pdf",
    "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260424_10816474.shtml",
    "https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml",
    "https://investor.szse.cn/lawrules/rule/trade/t20260424_620190.html",
    "https://docs.static.szse.cn/www/lawrules/rule/trade/current/W020260424690713155663.pdf",
]


def st_exit_limits(code, day, quote):
    if not re.fullmatch(r"(?:60\d{4}\.SH|00\d{4}\.SZ)", code) or quote["is_st"] != 1:
        return None
    # Rule review covers this delivery. Future regimes require an explicit update.
    if not "2019-01-01" <= day <= "2026-09-24":
        return None
    pre = quote["pre_close"]
    if not math.isfinite(pre) or pre <= 0:
        return None
    rate = .05 if day < "2026-07-06" else .10
    cents = math.floor(pre * 100 + .5)
    high = max(cents + 1, math.floor(cents * (1 + rate) + .5 + 1e-8)) / 100
    low = max(1, min(cents - 1, math.floor(cents * (1 - rate) + .5 + 1e-8))) / 100
    # Resumption/special sessions or inconsistent data are deliberately unsupported.
    if quote["high"] > high + 1e-6 or quote["low"] < low - 1e-6:
        return None
    return high, low
