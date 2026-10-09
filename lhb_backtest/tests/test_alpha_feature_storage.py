"""Shared keys preserve rows/reasons/clocks; mutated outputs and v1 mixing reject."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
from src.alpharesearch.feature_storage import KEYS,write_shared_index,read_shared_index,write_compact_candidate,read_compact_candidate
from src.alpharesearch.features.base import FeatureBlock
from src.alpharesearch.contracts import MissingReason as M
from src.alpharesearch.batch import AlphaBatchSpec
from src.researchops.alpha_experiments import verify_completed
from src.researchops.alpha_output_audit import audit_output
from src.technical.artifacts import digest,write_json
from tests.test_alpha_batch_registration import setup


def toy():
    keys=pd.DataFrame(dict(sample_id=["a"*64,"b"*64,"c"*64],trade_date=["2024-01-03"]*3,stock_code=["600001","600002","600003"]))
    values=keys.copy();values["score"]=[.2,np.nan,-.4]
    values["observed_end"]=pd.Timestamp("2024-01-03T15:00:00+08:00");values["known_at"]=pd.Timestamp("2024-01-03T15:30:00+08:00")
    missing=keys.copy();missing["score"]=[M.PRESENT.value,M.UNCOVERED.value,M.PRESENT.value]
    return FeatureBlock(values,missing,{"score":"ratio"},dict(key_columns=KEYS,source_id="toy",version_id="toy_v1",vintage="market_observation")).validate()


def stored(tmp_path):
    b=toy();index=write_shared_index(tmp_path,b.values[KEYS],max_bytes=1_000_000)
    write_compact_candidate(b,index,"candidate",max_bytes=1_000_000)
    return b,index


def test_lossless_roundtrip_unknown_and_clocks(tmp_path):
    b,_=stored(tmp_path);index=read_shared_index(tmp_path,max_bytes=1_000_000)
    result=read_compact_candidate(index,"candidate",max_bytes=1_000_000)
    pd.testing.assert_frame_equal(b.values,result.values);pd.testing.assert_frame_equal(b.missing,result.missing)
    assert result.units==b.units and result.metadata==b.metadata
    assert list(pd.read_parquet(tmp_path/"candidate/values.parquet"))==["row_index","score","observed_end","known_at"]
    assert not any(k in pd.read_parquet(tmp_path/"candidate/missing.parquet") for k in KEYS)


@pytest.mark.parametrize("mutation",["drop_row","value_order","reason_order","extra_column","wrong_reason","wrong_units","storage_owner","extra_file"])
def test_compact_mutations_reject(tmp_path,mutation):
    _,index=stored(tmp_path);target=tmp_path/"candidate"
    if mutation in ("drop_row","value_order","extra_column"):
        p=target/"values.parquet";v=pd.read_parquet(p)
        if mutation=="drop_row":v=v.iloc[:2]
        if mutation=="value_order":v=v.iloc[::-1]
        if mutation=="extra_column":v["future_label"]=1.
        v.to_parquet(p,index=False)
    if mutation in ("reason_order","wrong_reason"):
        p=target/"missing.parquet";v=pd.read_parquet(p)
        if mutation=="reason_order":v=v.iloc[::-1]
        else:v.loc[1,"score"]="present"
        v.to_parquet(p,index=False)
    if mutation=="wrong_units":
        p=target/"definition.json";v=json.loads(p.read_text());v["units"]={"score":"count"};write_json(p,v)
    if mutation=="storage_owner":
        p=target/"storage.json";v=json.loads(p.read_text());v["candidate_id"]="other";write_json(p,v)
    if mutation=="extra_file":(target/"unregistered.txt").write_text("extra")
    with pytest.raises(ValueError):read_compact_candidate(index,"candidate",max_bytes=1_000_000)


def test_shared_index_changed_rejects_before_parquet_parse(tmp_path):
    stored(tmp_path);p=tmp_path/"shared_keys.parquet";raw=p.read_bytes();p.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    with pytest.raises(ValueError,match="identity"):read_shared_index(tmp_path,max_bytes=1_000_000)


def test_shared_duplicate_keys_reject(tmp_path):
    b=toy();k=b.values[KEYS].copy();k.loc[1,"sample_id"]=k.loc[0,"sample_id"]
    with pytest.raises(ValueError,match="Unique"):write_shared_index(tmp_path,k,max_bytes=1_000_000)


def test_tiny_output_budget_retains_partial(tmp_path):
    with pytest.raises(ValueError,match="budget"):write_shared_index(tmp_path,toy().values[KEYS],max_bytes=1024)
    assert (tmp_path/"shared_keys.parquet.partial").exists()
    with pytest.raises(ValueError,match="Partial"):write_shared_index(tmp_path,toy().values[KEYS],max_bytes=1_000_000)


def test_existing_index_and_candidate_never_overwritten(tmp_path):
    b,index=stored(tmp_path)
    with pytest.raises(ValueError,match="already exists"):write_shared_index(tmp_path,b.values[KEYS],max_bytes=1_000_000)
    with pytest.raises(FileExistsError):write_compact_candidate(b,index,"candidate",max_bytes=1_000_000)


@pytest.fixture(scope="module")
def compact_registered(tmp_path_factory):
    ops,session,p,_,_=setup(tmp_path_factory.mktemp("compact_registered"));p["spec"]["schema"]="alpha-feature-batch-v2";p["spec"]["storage_layout"]="shared_index_v1"
    p["spec"]["candidates"].append(dict(copy.deepcopy(p["spec"]["candidates"][0]),candidate_id="identical_counted"))
    ex=ops.register(session,p);final=ops.execute(session,ex["id"])
    assert final["status"]=="completed"
    return ops,session,final,ops.root/"worker_jobs"/ex["id"]


def test_registered_compact_hand_values_and_full_audit(compact_registered):
    ops,session,ex,job=compact_registered;audit=verify_completed(ops,ex)
    assert audit["completed_candidates"]==2 and audit["fits"]==audit["accounts"]==0
    result=json.loads((job/"alpha_result/result.json").read_text());assert result["schema"]=="registered-alpha-feature-result-v2" and result["storage_layout"]=="shared_index_v1"
    index=read_shared_index(job/"alpha_result",max_bytes=10_000_000)
    for cid in ("relative_close_rank","identical_counted"):
        block=read_compact_candidate(index,cid,max_bytes=10_000_000)
        assert block.values.score.tolist()==[.75]*4
        assert block.values.trade_date.tolist()==["2024-01-03"]*2+["2024-01-04"]*2
    with pytest.raises(ValueError,match="already started"):ops.execute(session,ex["id"])


@pytest.mark.parametrize("mutation",["value_order","index_binding","missing_reason","shared_membership_order"])
def test_independent_compact_audit_rejects_resealed_mutations(compact_registered,mutation):
    ops,_,ex,job=compact_registered;folder=job/"alpha_result";target=folder/"relative_close_rank"
    originals={p.relative_to(folder):p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    try:
        if mutation=="value_order":
            p=target/"values.parquet";v=pd.read_parquet(p).iloc[::-1];v.to_parquet(p,index=False)
        if mutation=="index_binding":
            p=target/"storage.json";v=json.loads(p.read_text());v["keys_sha256"]="f"*64;write_json(p,v)
        if mutation=="missing_reason":
            p=target/"missing.parquet";v=pd.read_parquet(p);v["score"]="source_missing";v.to_parquet(p,index=False)
        if mutation=="shared_membership_order":
            p=folder/"shared_keys.parquet";v=pd.read_parquet(p);v.iloc[[0,1]]=v.iloc[[1,0]].to_numpy();v.to_parquet(p,index=False,compression="zstd")
            p=folder/"shared_keys.json";info=json.loads(p.read_text());info["keys_sha256"]=digest(folder/"shared_keys.parquet");info["keys_bytes"]=(folder/"shared_keys.parquet").stat().st_size;write_json(p,info)
            for cid in ("relative_close_rank","identical_counted"):
                p=folder/cid/"storage.json";rec=json.loads(p.read_text());rec["keys_sha256"]=info["keys_sha256"];rec["index_sha256"]=digest(folder/"shared_keys.json");write_json(p,rec)
        manifest=json.loads((folder/"manifest.json").read_text());manifest["artifacts"]={name:digest(folder/name) for name in manifest["artifacts"]};write_json(folder/"manifest.json",manifest)
        with pytest.raises(ValueError):audit_output(ops,ex)
    finally:
        for name,raw in originals.items():(folder/name).write_bytes(raw)


@pytest.mark.parametrize("mutation",["v2_no_layout","v2_bad_layout","v1_extra_layout"])
def test_versioned_storage_protocol_is_strict(tmp_path,mutation):
    _,_,p,registry,_=setup(tmp_path);spec=copy.deepcopy(p["spec"])
    if mutation.startswith("v2"):spec["schema"]="alpha-feature-batch-v2"
    if mutation!="v2_no_layout":spec["storage_layout"]="bad" if mutation=="v2_bad_layout" else "shared_index_v1"
    with pytest.raises(ValueError):AlphaBatchSpec.from_dict(spec,registry)
