# backend/utils.py
import math
import numpy as np
import pandas as pd

def cagr(start_value, end_value, periods):
    try:
        start_value = float(start_value)
        end_value = float(end_value)
        periods = float(periods)
    except Exception:
        return None
    if start_value <= 0 or periods <= 0:
        return None
    return (end_value / start_value) ** (1.0 / periods) - 1

def pct_change(old, new):
    try:
        old = float(old)
        new = float(new)
        return (new - old) / old
    except Exception:
        return None

def parse_percent(s):
    if s is None:
        return None
    try:
        s = str(s).strip()
        if s.endswith('%'):
            return float(s[:-1].replace(',',''))/100.0
        return float(s.replace(',',''))
    except Exception:
        return None

def avg_return_from_list(returns):
    arr = [r for r in returns if r is not None]
    if not arr:
        return None
    return float(np.mean(arr))

def sharpe_ratio(returns, risk_free_rate=0.0):
    import numpy as np
    arr = np.array([r for r in returns if r is not None])
    if arr.size == 0:
        return None
    excess = arr - risk_free_rate
    mean = excess.mean()
    std = excess.std(ddof=0)
    if std == 0:
        return None
    return mean / std

def compute_macaulay_duration(cashflows, yields):
    try:
        y = float(yields)
        pv_weights = []
        pv_total = 0.0
        for t, cf in cashflows:
            pv = cf / ((1+y)**t)
            pv_weights.append((t, pv))
            pv_total += pv
        if pv_total == 0:
            return None
        macaulay = sum(t * pv for t, pv in pv_weights) / pv_total
        return macaulay
    except Exception:
        return None
