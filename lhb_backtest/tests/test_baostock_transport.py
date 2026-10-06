from types import SimpleNamespace
import pytest
import compat.sitecustomize as compatibility

def test_closed_baostock_socket_raises_instead_of_spinning():
    wrapped=compatibility._CheckedSocket(SimpleNamespace(recv=lambda n:b''))
    with pytest.raises(ConnectionError,match='closed'):
        wrapped.recv(8192)

def test_missing_response_cannot_end_pagination_successfully(monkeypatch):
    monkeypatch.setattr(compatibility,'_original_send',lambda message:None)
    with pytest.raises(ConnectionError,match='no complete response'):
        compatibility._checked_send('request')

def test_full_page_without_final_response_is_rejected():
    from baostock.common.contants import BAOSTOCK_PER_PAGE_COUNT
    result=SimpleNamespace(data=[['value']]*BAOSTOCK_PER_PAGE_COUNT,fields=['name'],error_code='0',next=lambda:False)
    with pytest.raises(RuntimeError,match='without a final page'):
        compatibility.collect_result_data(result)
