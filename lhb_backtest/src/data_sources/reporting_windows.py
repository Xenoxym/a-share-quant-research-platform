"""Reporting periods are part of the LHB data key, not a trading signal."""
import re

_WINDOW = re.compile(r'(?:连续|最近)(\d+|[一二三四五六七八九十]+)(?:个)?(?:有成交的)?交易日')


def reporting_window_days(reason):
    if not isinstance(reason,str) or not reason.strip():
        return 0  # unknown must not silently become a single-day report
    values=[]
    for token in _WINDOW.findall(reason):
        if token.isdigit():
            values.append(int(token))
        else:
            numbers={'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
            if '十' in token:
                tens,ones=token.split('十',1)
                values.append(numbers.get(tens,1)*10+numbers.get(ones,0))
            else:
                values.append(numbers.get(token,0))
    if values:
        return max(values)
    if '异常期间' in reason:
        return 0
    return 1
