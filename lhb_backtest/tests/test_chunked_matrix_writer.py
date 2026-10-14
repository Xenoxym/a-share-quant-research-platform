"""Bounded matrix writes: independent expected rows, global keys and final checks."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from tests.test_matrix_materialization import assembly
from src.alpharesearch.assembly import FeatureAssembly
from src.alpharesearch.features.base import FeatureBlock
from src.researchops.matrix_materialization import save_learning_matrix_parts


def chunks(source,size):
    for start in range(0,len(source.block.values),size):
        b=source.block
        yield FeatureAssembly(FeatureBlock(b.values.iloc[start:start+size].reset_index(drop=True).copy(),
            b.missing.iloc[start:start+size].reset_index(drop=True).copy(),dict(b.units),deepcopy(b.metadata)),
            source.registry,source.source_bindings)


@pytest.mark.parametrize('size',[1,2,3,8])
def test_roundtrip_is_exact_and_row_groups_bounded(tmp_path,size,monkeypatch):
    source=assembly();folder=tmp_path/'matrix'
    def forbidden(*a,**k):raise AssertionError('Full pandas writer must not be used')
    monkeypatch.setattr(pd.DataFrame,'to_parquet',forbidden)
    result=save_learning_matrix_parts(chunks(source,size),folder,max_output_bytes=1000000,expected_rows=4)
    pd.testing.assert_frame_equal(pd.read_parquet(folder/'assembly_values.parquet'),source.block.values)
    pd.testing.assert_frame_equal(pd.read_parquet(folder/'assembly_missing.parquet'),source.block.missing)
    assert result['rows']==4
    parquet=pq.ParquetFile(folder/'assembly_values.parquet')
    assert max(parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups))<=size
    assert json.loads((folder/'receipt.json').read_text())['rows']==4


def test_cross_block_duplicate_rejected_without_receipt(tmp_path):
    source=assembly();parts=list(chunks(source,2));parts[1]=parts[0]
    with pytest.raises(ValueError,match='Duplicate matrix keys'):
        save_learning_matrix_parts(parts,tmp_path/'partial',max_output_bytes=1000000,expected_rows=4)
    assert not (tmp_path/'partial/receipt.json').exists()


@pytest.mark.parametrize('fault',['metadata','units','too_short','too_long','final_check'])
def test_changed_declaration_or_incomplete_source_has_no_receipt(tmp_path,fault):
    source=assembly();parts=list(chunks(source,2))
    if fault=='metadata':parts[1].block.metadata['source_id']='changed'
    if fault=='units':parts[1].block.units['stock.p']='cny'
    if fault=='too_short':parts=parts[:1]
    if fault=='too_long':parts=list(chunks(source,3))
    def stream():
        yield from parts
        if fault=='final_check':raise ValueError('source changed during consumption')
    with pytest.raises(ValueError):
        save_learning_matrix_parts(stream(),tmp_path/'partial',max_output_bytes=1000000,expected_rows=3 if fault=='too_long' else 4)
    assert not (tmp_path/'partial/receipt.json').exists()


def test_empty_source_has_no_directory(tmp_path):
    with pytest.raises(ValueError,match='At least one'):
        save_learning_matrix_parts(iter([]),tmp_path/'none',max_output_bytes=1000000)
    assert not (tmp_path/'none').exists()
