"""Independent source roles plus real protocol dispatch, without model calls."""
import copy
import hashlib
from pathlib import Path
import pytest
from .test_alpha_portfolio import scored_example,build,resign_scores
from src.alpharesearch.account import _code_bundle
from src.alpharesearch.score_reuse import reuse_code_bundle,validate_reuse_bundle
from src.researchops.account_experiments import protocol,upstream
from src.technical.artifacts import content_id

APP=Path(__file__).resolve().parents[1]


def identity_for(learning):
    files={n:dict(sha256=h,text=(APP/n).read_bytes().replace(b'\r\n',b'\n').decode('utf8')) for n,h in learning.receipt['fit_identity']['implementation_hashes'].items()}
    v=dict(schema='completed-learning-identity-v1',verification_mode='reuse_completed_identity_v1',producer_schema='registered-alpha-learn-v3',fit_id=learning.receipt['fit_id'],prediction_id=learning.receipt['prediction_id'],learning_receipt_id=content_id(learning.receipt),prediction_outputs=learning.receipt['prediction_outputs'],producer_files=files,additional_model_fits=0,additional_predict_calls=0,numeric_model_reproduction=False)
    return dict(v,identity_id=content_id(v))


def test_historical_model_source_can_differ_without_rewriting_scores():
    data=scored_example();learning=data[0]
    original=learning.predictions.copy(deep=True)
    identity=identity_for(learning)
    n='src/alpharesearch/models.py';old=identity['producer_files'][n]['text']+'\n# historic supported model implementation\n'
    sha=hashlib.sha256(old.encode()).hexdigest();identity['producer_files'][n]=dict(text=old,sha256=sha)
    learning.receipt['fit_identity']['implementation_hashes'][n]=sha
    learning.receipt['fitted_model']['implementation_hashes'][n]=sha
    # Fixture simulates a previously audited source; production requires DB audit.
    learning.receipt['fit_id']=content_id(learning.receipt['fit_identity']);learning.predictions['fit_id']=learning.receipt['fit_id']
    resign_scores(learning,membership=True)
    identity.update(fit_id=learning.receipt['fit_id'],prediction_id=learning.receipt['prediction_id'],learning_receipt_id=content_id(learning.receipt),prediction_outputs=learning.receipt['prediction_outputs'])
    identity['identity_id']=content_id({k:v for k,v in identity.items() if k!='identity_id'})
    portfolio=build(data)
    with pytest.raises(ValueError,match='installed implementation'):_code_bundle(learning,portfolio)
    bundle=reuse_code_bundle(learning,portfolio,identity)
    assert bundle['producer_identity']['producer_files'][n]['sha256']==sha
    assert learning.predictions.score.equals(original.score)
    assert portfolio.decisions.selected.sum()==4
    validate_reuse_bundle(bundle,learning,portfolio)


@pytest.mark.parametrize('change',['producer_text','producer_hash','consumer_text','consumer_hash','shared_contract','fit_id','schema','identity_id','receipt_id','prediction_outputs'])
def test_independent_roles_and_identity_reject_tampering(change):
    data=scored_example();portfolio=build(data);bundle=reuse_code_bundle(data[0],portfolio,identity_for(data[0]));identity=bundle['producer_identity']
    if change=='producer_text':next(iter(identity['producer_files'].values()))['text']+='changed'
    elif change=='producer_hash':next(iter(identity['producer_files'].values()))['sha256']='0'*64
    elif change=='consumer_text':next(iter(bundle['consumer_files'].values()))['text']+='changed'
    elif change=='consumer_hash':next(iter(bundle['consumer_files'].values()))['sha256']='0'*64
    elif change=='shared_contract':
        n='src/alpharesearch/contracts.py';new=bundle['consumer_files'][n]['text']+'\n# changed contract\n';bundle['consumer_files'][n]=dict(text=new,sha256=hashlib.sha256(new.encode()).hexdigest())
    elif change=='fit_id':identity['fit_id']='0'*64
    elif change=='schema':identity['producer_schema']='unknown'
    elif change=='identity_id':identity['identity_id']='0'*64
    elif change=='receipt_id':identity['learning_receipt_id']='0'*64
    else:identity['prediction_outputs']=dict(rows=999)
    with pytest.raises(ValueError):validate_reuse_bundle(bundle,data[0],portfolio)


def test_legacy_same_version_bundle_remains_v1():
    data=scored_example();assert _code_bundle(data[0],build(data))['schema']=='score-account-code-bundle-v1'


def test_score_structure_and_clock_are_still_checked():
    data=scored_example();data[0].predictions.loc[0,'execution_at']=data[0].predictions.loc[0,'decision_at'];resign_scores(data[0],membership=True)
    with pytest.raises(ValueError):build(data)

@pytest.fixture(autouse=True)
def forbid_learning_calls(monkeypatch):
    import joblib
    from sklearn.linear_model import Ridge,ElasticNet,HuberRegressor
    from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    def forbidden(*args,**kwargs):pytest.fail('Score reuse must not fit, predict or load checkpoint')
    for cls in [Ridge,ElasticNet,HuberRegressor,RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor,MLPRegressor]:
        monkeypatch.setattr(cls,'fit',forbidden);monkeypatch.setattr(cls,'predict',forbidden)
    monkeypatch.setattr(joblib,'load',forbidden)


def account_protocol(data,schema='registered-alpha-account-v2'):
    from dataclasses import replace
    from src.technical.contracts import ResearchSpec,Strategy,Experiment
    cfg=data[-1].to_dict()
    account=ResearchSpec(strategy=replace(Strategy(),name=cfg['name'],top_k=cfg['top_k'],allocation=cfg['allocation'],rebalance=cfg['rebalance']),experiment=Experiment(cfg['start'],cfg['end']))
    source=dict(kind='registered_learning',experiment_id='exp-'+'a'*12)
    if schema.endswith('v2'):source['verification_mode']='reuse_completed_identity_v1'
    return dict(schema=schema,portfolio=cfg,account=account.to_dict(),scenarios=['zero_transaction_cost','configured'],snapshot_id='a'*24,snapshot_manifest_hash='b'*64,score_source=source,budget=dict(max_input_bytes=1000000,max_task_account_intents=3,max_real_account_intents=2,max_synthetic_account_intents=1))


@pytest.mark.parametrize('schema',['registered-alpha-account-v1','registered-alpha-account-v2'])
def test_explicit_versioned_protocol_preserves_account_classification(schema):
    from src.researchops.account_intents import count
    doc=account_protocol(scored_example(),schema)
    _,_,parsed=protocol(doc)
    assert parsed['schema']==schema
    assert count({'kind':'alpha_account','spec':parsed,'planned_accounts':2})==(2,0)


@pytest.mark.parametrize('change',['v1_hidden_mode','v2_unknown_mode','v2_missing_mode','v2_synthetic','unknown_schema'])
def test_reuse_mode_cannot_be_implicit_or_synthetic(change):
    doc=account_protocol(scored_example())
    protocol(doc)  # Prove the unmodified fixture is valid before negative mutation.
    if change=='v1_hidden_mode':doc['schema']='registered-alpha-account-v1'
    elif change=='v2_unknown_mode':doc['score_source']['verification_mode']='trust_success'
    elif change=='v2_missing_mode':doc['score_source'].pop('verification_mode')
    elif change=='v2_synthetic':doc['score_source']=dict(kind='synthetic_stub',statement='engineering scores only; no fitted strategy')
    else:doc['schema']='unknown'
    with pytest.raises(ValueError):protocol(doc)


def test_worker_rechecks_cached_source_using_readonly_database(monkeypatch):
    import src.researchops.account_worker as worker
    import src.researchops.service as service
    calls=[];source=dict(reuse_identity=dict(identity_id='certificate'))
    cfg=dict(project='p',root='r',spec=dict(schema='registered-alpha-account-v2'),upstream=copy.deepcopy(source))
    def readonly(project,root,readonly):
        calls.append((project,root,readonly));return object()
    monkeypatch.setattr(service,'Research',readonly);monkeypatch.setattr(worker,'upstream',lambda *a:source)
    assert worker.score_reuse_identity(cfg)==source['reuse_identity']
    assert calls==[('p','r',True)]
    cfg['upstream']['reuse_identity']['identity_id']='changed'
    with pytest.raises(ValueError,match='producer changed'):worker.score_reuse_identity(cfg)


def test_legacy_worker_does_not_enter_new_reuse_policy(monkeypatch):
    import src.researchops.account_worker as worker
    monkeypatch.setattr(worker,'upstream',lambda *a:pytest.fail('legacy must not use identity mode'))
    assert worker.score_reuse_identity(dict(spec=dict(schema='registered-alpha-account-v1')))==None
