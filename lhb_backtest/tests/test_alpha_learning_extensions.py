"""Hand fixtures only; exactlysix syntheticfit intents, no market/account work."""
from pathlib import Path
import dataclasses, hashlib, json
import numpy as np
import pandas as pd
import pytest
from src.alpharesearch.models import TabularRegressor, model_spec
from src.technical.artifacts import content_id

from src.alpharesearch import targets, checkpoints


def fixture():
    d=[];l=[]
    for day,values in [("2024-01-05",[.01,.02,.03]),("2024-01-12",[.02,.02,.04])]:
        entry=pd.Timestamp(day+"T09:30:00+08:00")+pd.Timedelta(days=3)
        end=entry+pd.Timedelta(days=7)
        for i,y in enumerate(values):
            keys=dict(sample_id=day+"_"+str(i),trade_date=day,stock_code=f"60000{i}.SH")
            d.append(dict(**keys,decision_at=day+"T15:30:00+08:00",execution_at=entry,target_end_at=end))
            l.append(dict(**keys,target_return=y,label_start=entry,label_end=end,label_known_at=end,label_observed=True))
    return pd.DataFrame(d),pd.DataFrame(l)


def spec(kind="raw_return",**kwargs):
    return dataclasses.replace(targets.TrainingTargetSpec(kind,"2024-01-05","2024-01-12","2024-01-22T15:30:00+08:00"),**kwargs)


@pytest.mark.parametrize("kind,expected",[("raw_return",[.01,.02,.03]),("date_centered",[-.01,0,.01]),("date_zscore",[-np.sqrt(1.5),0,np.sqrt(1.5)]),("date_rank",[-.5,0,.5])])
def test_hand_values_and_units(kind,expected):
    d,l=fixture();r=targets.build_training_targets(d,l,spec(kind))
    np.testing.assert_allclose(r.rows.training_target.iloc[:3],expected,atol=1e-14)
    assert r.receipt['usable_rows']==6 and not r.receipt['prediction_labels_used']
    np.testing.assert_allclose(r.rows.raw_target,l.target_return)


def test_rank_average_ties():
    d,l=fixture();r=targets.build_training_targets(d,l,spec('date_rank'))
    np.testing.assert_allclose(r.rows.training_target.iloc[3:],[-.25,-.25,.5])


def test_one_ns_maturity_boundary():
    d,l=fixture();l['label_known_at']=l['label_known_at'].astype(pd.DatetimeTZDtype(unit='ns',tz=l['label_known_at'].dtype.tz));cut=pd.Timestamp(spec().fit_cutoff);l.loc[0,'label_known_at']=cut+pd.Timedelta(nanoseconds=1)
    r=targets.build_training_targets(d,l,spec());assert len(r.rows)==6 and r.rows.target_reason.iloc[0]=='label_immature' and pd.isna(r.rows.raw_target.iloc[0])
    l.loc[0,'label_known_at']=cut;r=targets.build_training_targets(d,l,spec());assert r.rows.target_reason.iloc[0]=='present'


def test_unknown_label_keeps_decision_not_zero():
    d,l=fixture();l.loc[0,['target_return','label_known_at']]=[np.nan,pd.NaT];l.loc[0,'label_observed']=False
    r=targets.build_training_targets(d,l,spec());assert len(r.rows)==6 and r.rows.target_reason.iloc[0]=='label_unobserved' and pd.isna(r.rows.training_target.iloc[0])


def test_absent_label_retains_key():
    d,l=fixture();r=targets.build_training_targets(d,l.iloc[1:],spec());assert r.rows.target_reason.iloc[0]=='label_absent' and len(r.rows)==6


def test_future_numeric_label_not_read():
    d,l=fixture();r=targets.build_training_targets(d,l,spec())
    d2,l2=fixture();date_map={'2024-01-05':'2024-02-02','2024-01-12':'2024-02-09'}
    d2['trade_date']=d2['trade_date'].map(date_map);l2['trade_date']=l2['trade_date'].map(date_map)
    d2['sample_id']+='future';l2['sample_id']+='future'
    d2['decision_at']=pd.to_datetime(d2['decision_at'])+pd.Timedelta(days=28)
    for clock in ('execution_at','target_end_at'):d2[clock]=d2[clock]+pd.Timedelta(days=28)
    for clock in ('label_start','label_end','label_known_at'):l2[clock]=l2[clock]+pd.Timedelta(days=28)
    l2['target_return']='unusablefuturevalue'
    actual=targets.build_training_targets(pd.concat([d,d2],ignore_index=True),pd.concat([l,l2],ignore_index=True),spec())
    pd.testing.assert_frame_equal(r.rows,actual.rows);assert r.receipt==actual.receipt


def test_mature_only_date_denominator():
    d,l=fixture();l.loc[0,'label_known_at']=pd.Timestamp('2024-02-01T09:30:00+08:00')
    r=targets.build_training_targets(d,l,spec('date_centered'));np.testing.assert_allclose(r.rows.training_target.iloc[1:3],[-.005,.005]);assert r.rows.mature_date_rows.iloc[1]==2


@pytest.mark.parametrize('kind',['date_zscore','date_rank'])
def test_constant_group_masks(kind):
    d,l=fixture();l.loc[:2,'target_return']=.01;r=targets.build_training_targets(d,l,spec(kind));assert r.rows.target_reason.iloc[:3].eq('constant_date_group').all() and r.rows.training_target.iloc[:3].isna().all()


def test_thin_group_masks():
    d,l=fixture();r=targets.build_training_targets(d,l,spec('date_centered',min_date_rows=4));assert r.rows.target_reason.eq('thin_date_group').all()


@pytest.mark.parametrize('fault',['naive_cutoff','known_before_end','label_interval','identity','observed_int','unknown_with_value','infinity','row_cap'])
def test_invalid_target_inputs(fault):
    d,l=fixture();cfg=spec()
    if fault=='naive_cutoff':cfg=spec(fit_cutoff='2024-01-22T15:30:00')
    elif fault=='known_before_end':l['label_known_at']=l['label_known_at'].astype(pd.DatetimeTZDtype(unit='ns',tz=l['label_known_at'].dtype.tz));l.loc[0,'label_known_at']=pd.Timestamp(l.loc[0,'label_end'])-pd.Timedelta(nanoseconds=1)
    elif fault=='label_interval':l.loc[0,'label_end']=pd.Timestamp(l.loc[0,'label_end'])+pd.Timedelta(seconds=1)
    elif fault=='identity':l.loc[0,'sample_id']='badid'
    elif fault=='observed_int':l['label_observed']=1
    elif fault=='unknown_with_value':l.loc[0,'label_observed']=False
    elif fault=='infinity':l.loc[0,'target_return']=np.inf
    else:cfg=spec(max_input_rows=2)
    with pytest.raises(ValueError):targets.build_training_targets(d,l,cfg)


@pytest.fixture(scope='module')
def fitted_models(tmp_path_factory):
    folder=tmp_path_factory.mktemp('six_fits'); ledger=[]
    x=np.linspace(-1,1,40);X=pd.DataFrame(dict(a=x,b=np.cos(x*3)));y=pd.Series(.02*x+.01*np.cos(x*3));w=pd.Series(np.linspace(.5,1.5,40));X.loc[2,'b']=np.nan
    models={};configs={'ridge':{},'elastic_net':{'max_iter':4000,'tol':1e-4},'huber':{'max_iter':1000},'random_forest':{'n_estimators':8,'max_depth':3,'min_samples_leaf':2},'extra_trees':{'n_estimators':8,'max_depth':3,'min_samples_leaf':2},'hist_gbdt':{'max_iter':8,'min_samples_leaf':2}}
    for name,params in configs.items():
        ledger.append(dict(family=name,state='intent'));(folder/'fit_intents.json').write_text(json.dumps(ledger),encoding='utf8')
        adapter=TabularRegressor(model_spec(name,params=params,max_train_rows=100,max_features=3,max_matrix_cells=1000)).fit(X,y,sample_weight=w)
        models[name]=adapter;ledger[-1]['state']='succeeded';(folder/'fit_intents.json').write_text(json.dumps(ledger),encoding='utf8')
    hashv=lambda v:hashlib.sha256(pd.util.hash_pandas_object(v,index=True).to_numpy().tobytes()).hexdigest()
    context=dict(fit_id=content_id(dict(X=hashv(X),Y=hashv(y),weights=hashv(w))),fit_cutoff='2024-01-22T15:30:00+08:00',training_inputs_sha256=hashv(X),training_targets_sha256=hashv(y),training_weights_sha256=hashv(w),target_spec_id=spec().target_id)
    return models,X,context


@pytest.mark.parametrize('family',['ridge','elastic_net','huber','random_forest','extra_trees','hist_gbdt'])
def test_checkpoint_roundtrip_without_fit(family,fitted_models,tmp_path):
    models,X,context=fitted_models;model=models[family];before=model.predict(X);r=checkpoints.save_checkpoint(model,tmp_path/family,context)
    after=checkpoints.load_checkpoint(tmp_path/family,expected_manifest_sha256=r['manifest_sha256'],trusted_local=True)
    np.testing.assert_array_equal(after.predict(X),before);assert after.fit_attempts==0 and after.successful_fits==0 and after.metadata()==model.metadata()
    with pytest.raises(ValueError):after.fit(X,pd.Series(np.ones(len(X))))


@pytest.mark.parametrize('fault',['untrusted','manifest','payload','cap','extra_file','overwrite','column_order'])
def test_checkpoint_rejects(fault,fitted_models,tmp_path):
    models,X,context=fitted_models;folder=tmp_path/'model';r=checkpoints.save_checkpoint(models['ridge'],folder,context)
    args=dict(expected_manifest_sha256=r['manifest_sha256'],trusted_local=True)
    if fault=='untrusted':args['trusted_local']=False
    elif fault=='manifest':(folder/'manifest.json').write_bytes((folder/'manifest.json').read_bytes()+b' ')
    elif fault=='payload':(folder/'model.joblib').write_bytes(b'corrupted')
    elif fault=='cap':args['max_bytes']=1024
    elif fault=='extra_file':(folder/'extra.txt').write_text('unexpected')
    elif fault=='overwrite':
        with pytest.raises(FileExistsError):checkpoints.save_checkpoint(models['ridge'],folder,context)
        return
    elif fault=='column_order':
        loaded=checkpoints.load_checkpoint(folder,**args)
        with pytest.raises(ValueError):loaded.predict(X[['b','a']])
        return
    with pytest.raises(ValueError):checkpoints.load_checkpoint(folder,**args)


@pytest.mark.parametrize('case',['at','after','interval_mismatch'])
def test_iso_string_nanosecond_boundaries(case):
    d,l=fixture()
    for clock in ('label_start','label_end','label_known_at'):
        l[clock]=l[clock].map(lambda x:x.isoformat()).astype(object)
    if case=='interval_mismatch':
        l.loc[0,'label_end']=(pd.Timestamp(l.loc[0,'label_end'])+pd.Timedelta(nanoseconds=1)).isoformat()
        with pytest.raises(ValueError):targets.build_training_targets(d,l,spec())
        return
    l.loc[0,'label_known_at']='2024-01-22T15:30:00.00000000'+('1' if case=='after' else '0')+'+08:00'
    result=targets.build_training_targets(d,l,spec())
    assert result.rows.target_reason.iloc[0]==('label_immature' if case=='after' else 'present')


def test_iso_cutoff_retains_one_nanosecond():
    d,l=fixture();clock='2024-01-22T15:30:00.000000001+08:00'
    l['label_known_at']=l['label_known_at'].astype(object);l.loc[0,'label_known_at']=clock
    result=targets.build_training_targets(d,l,spec(fit_cutoff=clock))
    assert result.rows.target_reason.iloc[0]=='present' and '.000000001' in result.receipt['spec']['fit_cutoff']
