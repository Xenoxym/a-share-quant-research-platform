import pandas as pd
import pytest

from src.data_sources.parquet_merge import merge_kline_file


def test_file_merge_preserves_history_and_known_cells(tmp_path):
    path = tmp_path/'daily.parquet'
    old = pd.DataFrame({'stock_code':['A','A'], 'trade_date':pd.to_datetime(['2026-09-21','2026-09-22']),
                        'close':[10.,11.], 'old_feature':[1.,2.]})
    old.to_parquet(path,index=False)
    new = pd.DataFrame({'stock_code':['A','A'], 'trade_date':['2026-09-22','2026-09-23'],
                       'close':[None,12.], 'high_limit':[12.1,13.2]})
    merge_kline_file(path,new)
    result = pd.read_parquet(path)
    assert result.trade_date.tolist()==['2026-09-21','2026-09-22','2026-09-23']
    assert result.close.tolist()==[10.,11.,12.]
    assert result.old_feature.iloc[:2].tolist()==[1.,2.]
    assert result.high_limit.iloc[1:].tolist()==[12.1,13.2]
    merge_kline_file(path,new)
    pd.testing.assert_frame_equal(result,pd.read_parquet(path))
    assert not list(tmp_path.glob('kline_merge_*'))


def test_invalid_file_merge_preserves_original_bytes(tmp_path):
    path=tmp_path/'daily.parquet'
    data=pd.DataFrame({'stock_code':['A'],'trade_date':['2026-09-22'],'close':[11.]})
    data.to_parquet(path,index=False)
    before=path.read_bytes()
    with pytest.raises(ValueError,match='duplicate keys'):
        merge_kline_file(path,pd.concat([data,data],ignore_index=True))
    assert path.read_bytes()==before
    assert sorted(p.name for p in tmp_path.iterdir())==['daily.parquet']
