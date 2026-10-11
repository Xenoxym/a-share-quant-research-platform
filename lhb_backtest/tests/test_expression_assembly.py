"""Synthetic expression handoff semantics; no registered DB certification or account."""
from copy import deepcopy
import weakref
import numpy as np
import pandas as pd
import pytest
from src.alpharesearch.dsl import Expr, compile_expression
from src.alpharesearch.features.base import FeatureBlock
from src.alpharesearch.registry import FeatureDefinition, FeatureRegistry
from src.alpharesearch.expression_assembly import assemble_declared_expression_blocks, assemble_registered_expression_batch
KEYS = ["sample_id","trade_date","stock_code"]

def case():
    p = pd.DataFrame(dict(sample_id=["b","a","c"],trade_date=["2022-12-23"]*2+["2022-12-30"],
        stock_code=["600002.SH","600001.SH","600001.SH"],
        decision_at=["2022-12-23T07:00:00+00:00"]*2+["2022-12-30T07:00:00+00:00"]))
    d = FeatureDefinition("stock.x","ratio","stock_day","Synthetic declared leaf",("synthetic.source",),"{}",
        (("tests/test_expression_assembly.py","a"*64),),"explicit decision clock","unknown remains missing")
    r = FeatureRegistry([d])
    compiled = compile_expression(Expr.ref("stock.x",d.definition_id),r)
    c = dict(candidate_id="synthetic_00",family="synthetic",hypothesis="Synthetic interface only",expression=compiled.to_dict())
    v = p[KEYS].copy(); v["score"]=[2.,np.nan,1.]; v["known_at"]=p.decision_at;v["observed_end"]=p.decision_at
    m = p[KEYS].copy();m["score"]=["present","source_uncovered","present"]
    meta = dict(schema="alpha-expression-values-v1",key_columns=KEYS,universe_id="synthetic_u",
        expression_id=compiled.expression_id,source_id="synthetic_not_registered",version_id="b"*64,vintage="market_observation",
        dependencies=[dict(key=d.key,definition_id=d.definition_id,materialization_id="c"*64,
            source_id="synthetic",version_id="d"*64,vintage="market_observation")])
    return p,r,c,FeatureBlock(v,m,{"score":"ratio"},meta).validate()

def assemble(p,r,items):
    return assemble_declared_expression_blocks(p,items,r,universe_id="synthetic_u",
        provenance={"schema":"synthetic_declared_only","registered_DB_authenticated":False},max_rows=3,max_matrix_cells=192)

def test_preserves_all_members_original_order_clocks_and_nonzero_missing():
    p,r,c,b=case(); result=assemble(p,r,[(c,b,{"values":"e"*64})])
    assert result.block.values[KEYS].equals(p[KEYS])
    assert result.block.missing[KEYS].equals(p[KEYS])
    assert result.block.values["expr.synthetic_00"].iloc[0]==2.
    assert np.isnan(result.block.values["expr.synthetic_00"].iloc[1])
    assert result.block.missing["expr.synthetic_00"].tolist()==["present","source_uncovered","present"]
    assert result.block.metadata["registered_batch_verified_by_wrapper"] is False

@pytest.mark.parametrize("mutation",["missing_member","row_order"])
def test_membership_cannot_be_reduced_or_reordered(mutation):
    p,r,c,b=case();assemble(p,r,[(c,b,{"values":"e"*64})])
    if mutation=="missing_member":
        b.values=b.values.iloc[:2].reset_index(drop=True);b.missing=b.missing.iloc[:2].reset_index(drop=True)
    else:
        b.values=b.values.iloc[[1,0,2]].reset_index(drop=True);b.missing=b.missing.iloc[[1,0,2]].reset_index(drop=True)
    with pytest.raises(ValueError,match="complete membership and row order"):
        assemble(p,r,[(c,b,{"values":"e"*64})])

@pytest.mark.parametrize("offset",[-1,1])
def test_different_or_future_clock_is_not_accepted(offset):
    p,r,c,b=case();assemble(p,r,[(c,b,{"values":"e"*64})])
    value=(pd.Timestamp(p.decision_at.iloc[0])+pd.Timedelta(minutes=offset)).isoformat()
    b.values.loc[0,["known_at","observed_end"]]=value
    with pytest.raises(ValueError,match="conservative decision clock"):
        assemble(p,r,[(c,b,{"values":"e"*64})])

@pytest.mark.parametrize("mutation",["duplicate_candidate","unit","universe","dependency","duplicate_dependency"])
def test_semantic_identity_mutations_rejected_after_valid_baseline(mutation):
    p,r,c,b=case();assemble(p,r,[(c,b,{"values":"e"*64})])
    items=[(c,b,{"values":"e"*64})]
    if mutation=="duplicate_candidate":items*=2
    elif mutation=="unit":b.units["score"]="shares"
    elif mutation=="universe":b.metadata["universe_id"]="different_u"
    elif mutation=="dependency":b.metadata["dependencies"][0]["definition_id"]="f"*64
    else:b.metadata["dependencies"].append(deepcopy(b.metadata["dependencies"][0]))
    with pytest.raises(ValueError,match="Duplicate expression|definition/unit/universe|dependency identities"):
        assemble(p,r,items)

def test_definition_stable_but_materialization_changes_with_output_SHA():
    p,r,c,b=case()
    first=assemble(p,r,[(c,b,{"values":"e"*64})])
    second=assemble(p,r,[(c,b,{"values":"f"*64})])
    assert first.registry.version_id==second.registry.version_id
    assert first.registry.definitions[0].definition_id==second.registry.definitions[0].definition_id
    assert first.source_bindings[0]["materialization_id"]!=second.source_bindings[0]["materialization_id"]
    assert first.block.metadata["version_id"]!=second.block.metadata["version_id"]

def test_generator_does_not_retain_all_64_FeatureBlocks():
    p,r,_,_=case(); refs=[]
    def items():
        for i in range(64):
            _,_,c,b=case();c["candidate_id"]="synthetic_"+str(i).zfill(2)
            refs.append(weakref.ref(b))
            assert sum(ref() is not None for ref in refs)<=2
            yield c,b,{"values":"e"*64}
            del b
    output=assemble(p,r,items())
    assert len(output.block.units)==64 and len(output.block.values)==3
    assert sum(ref() is not None for ref in refs)==0

def test_pure_fixture_cannot_claim_registered_wrapper_authentication():
    p,_,_,_=case()
    with pytest.raises(ValueError,match="Actual Research DB"):
        assemble_registered_expression_batch(object(),{"id":"synthetic_not_registered"},p,max_rows=3,max_matrix_cells=192)
