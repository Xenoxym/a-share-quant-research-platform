"""Narrow BaoStock compatibility for the official downloader under pandas >= 2.

BaoStock ResultData.get_data calls the removed DataFrame.append on page two.
Do not patch pandas globally; preserve the provider's paging protocol instead.
"""
import pandas as pd


def collect_result_data(result):
    if not result.data:
        return pd.DataFrame(columns=result.fields)
    frames = [pd.DataFrame(result.data, columns=result.fields)]
    result.cur_row_num = len(result.data)
    while result.error_code == '0':
        if not result.next():
            # Failed socket reads can leave error_code='0' and the old full
            # page in place. That is not a successful end of pagination.
            from baostock.common.contants import BAOSTOCK_PER_PAGE_COUNT
            if len(result.data) == BAOSTOCK_PER_PAGE_COUNT:
                raise RuntimeError('BaoStock pagination ended without a final page')
            break
        frames.append(pd.DataFrame(result.data, columns=result.fields))
        result.cur_row_num = len(result.data)
    if result.error_code != '0':
        raise RuntimeError(f'BaoStock pagination failed: {result.error_code}: {result.error_msg}')
    return pd.concat(frames, ignore_index=True)


try:
    from baostock.data.resultset import ResultData
except ImportError:
    pass
else:
    if not hasattr(pd.DataFrame, 'append'):
        ResultData.get_data = collect_result_data

    # BaoStock's receive loop does not handle recv(b''): a peer disconnect
    # becomes an endless busy loop. Keep its protocol, but fail on EOF and
    # missing responses instead of accepting a truncated page as completion.
    from baostock.common import context as _bs_context
    from baostock.util import socketutil as _bs_socket

    class _CheckedSocket:
        def __init__(self, raw):
            self.raw = raw

        def __getattr__(self, name):
            return getattr(self.raw, name)

        def recv(self, size):
            data = self.raw.recv(size)
            if not data:
                raise ConnectionError('BaoStock closed the connection before completing its response')
            return data

    _original_connect = _bs_socket.SocketUtil.connect
    _original_send = _bs_socket.send_msg

    def _checked_connect(self, *args, **kwargs):
        _original_connect(self, *args, **kwargs)
        raw = getattr(_bs_context, 'default_socket', None)
        if raw is not None and not isinstance(raw, _CheckedSocket):
            raw.settimeout(30)
            _bs_context.default_socket = _CheckedSocket(raw)

    def _checked_send(msg):
        result = _original_send(msg)
        if not result:
            raise ConnectionError('BaoStock returned no complete response')
        return result

    _bs_socket.SocketUtil.connect = _checked_connect
    _bs_socket.send_msg = _checked_send
