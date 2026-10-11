"""Native v3 evidence with deliberate blocked/partial/fee-fork cases; no model fit."""
from dataclasses import replace
import hashlib,json,shutil
import pandas as pd
import pytest
from .test_alpha_account_audit import inputs,resign_archive
from .test_score_reuse_versions import identity_for
from src.alpharesearch.holding_policy import ActualHoldingPolicy
from src.alpharesearch.account import prepare_score_account
from src.alpharesearch.audit import audit_account_tables,audit_score_account_run
from src.technical.portfolio import simulate
from src.technical.runner import run
from src.technical.artifacts import digest,write_json
from src.researchops.account_experiments import protocol

@pytest.fixture(name='tmp_path')
def stable_path(tmp_path_factory,request):
    p=tmp_path_factory.getbasetemp()/('h_'+hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]);p.mkdir();return p

def prepared_case(root,case='regular'):
    data,snapshot,spec=inputs(root)
    if case=='fees_fork':
        spec=replace(spec,execution=replace(spec.execution,initial_cash=2200.))
        frame=pd.read_parquet(snapshot/'status.parquet')
        # First-session fee cash reserve permits one vs two fills. Block the
        # otherwise valid daily catch-up through the next signal close.
        idx=frame.status_type.eq('HALT') & frame.trade_date.between('2024-01-09','2024-01-12')
        frame.loc[idx,'symbols']=frame.loc[idx,'symbols'].map(lambda xs:sorted(set(list(xs)+['600002.SH'])))
        frame.to_parquet(snapshot/'status.parquet',index=False)
        sm=json.loads((snapshot/'manifest.json').read_text());sm['artifacts']['status.parquet']=digest(snapshot/'status.parquet')
        write_json(snapshot/'manifest.json',sm)
    if case in ('blocked_exit','partial_exit','execution_action'):
        name='bars' if case!='execution_action' else 'actions'
        frame=pd.read_parquet(snapshot/(name+'.parquet'))
        if case=='execution_action':
            frame=pd.concat([frame,pd.DataFrame([dict(stock_code='600002.SH',trade_date='2024-01-15',
                allotted_ps=.1,rationed_ps=0.,rationed_px=0.,bonus_ps=0.,dividend=.8)])],ignore_index=True)
        else:
            idx=frame.stock_code.eq('600003.SH') & frame.trade_date.ge('2024-01-15')
            if case=='blocked_exit':frame.loc[idx,'low_limit']=frame.loc[idx,'open'].to_numpy()
            else:frame.loc[idx,'volume']=2000. # 5% cap100 leaves residual odd lot occupying its slot.
        frame.to_parquet(snapshot/(name+'.parquet'),index=False)
        sm=json.loads((snapshot/'manifest.json').read_text());sm['counts'][name]=len(frame)
        sm['artifacts'][name+'.parquet']=digest(snapshot/(name+'.parquet'));write_json(snapshot/'manifest.json',sm)
    hp=ActualHoldingPolicy(2,2,3,.98)
    prep=prepare_score_account(*data,snapshot,spec,reuse_identity=identity_for(data[0]),holding_policy=hp)
    return data,snapshot,spec,hp,prep

@pytest.mark.parametrize('case',['regular','blocked_exit','partial_exit','fees_fork','execution_action'])
def test_native_actual_selection_reconciles_and_preserves_residual_slots(case,tmp_path):
    data,snapshot,spec,hp,p=prepared_case(tmp_path,case)
    sessions=[d for d in data[2] if spec.experiment.start_date<=d<=spec.experiment.end_date]
    tables={n:pd.read_parquet(snapshot/(n+'.parquet')) for n in ('actions','status','metadata')}
    outputs={}
    for scenario in ('zero_transaction_cost','configured'):
        out=simulate(p['decisions'],p['schedule'],p['market'],sessions,spec,scenario,holding_policy=hp)
        audit=audit_account_tables(out,pd.read_parquet(snapshot/'bars.parquet'),spec,scenario,sessions,
             decisions=p['decisions'],schedule=p['schedule'],**tables,holding_policy=hp.to_dict(),calendar=data[2])
        assert audit['actual_holding_policy']['primitive_ledger_state_reconstruction']
        assert out['equity'].positions.max()<=2
        first=json.loads(out['holding_choices'].selection_json.iloc[0]);second=json.loads(out['holding_choices'].selection_json.iloc[1])
        assert all(t['reason']=='new_entry' for t in first['targets'])
        state=json.loads(out['holding_choices'].state_json.iloc[1]);assert state['asof']=='2024-01-12'
        if case in ('blocked_exit','partial_exit'):
            assert '600003.SH' in set(out['positions'].loc[out['positions'].date.eq('2024-01-15'),'stock_code'])
            assert not (out['fills'].stock_code.eq('600000.SH') & out['fills'].date.eq('2024-01-15')).any()
            assert out['orders'].reason.eq('position_limit_with_pending_exits').any()
        if case=='execution_action':
            held=next(h for h in state['holdings'] if h['stock_code']=='600002.SH')
            assert held['mark']==10. and held['quote_date']=='2024-01-12'
        outputs[scenario]=out
    if case=='fees_fork':
        a,b=[json.loads(outputs[s]['holding_choices'].state_json.iloc[1]) for s in outputs]
        assert {x['stock_code'] for x in a['holdings']}!={x['stock_code'] for x in b['holdings']}
        a,b=[json.loads(outputs[s]['holding_choices'].selection_json.iloc[1])['targets'] for s in outputs]
        assert {x['stock_code'] for x in a}!={x['stock_code'] for x in b}

@pytest.fixture(scope='module')
def archive(tmp_path_factory):
    root=tmp_path_factory.getbasetemp()/'holdarc';root.mkdir()
    data,snapshot,spec,hp,prepared=prepared_case(root)
    folder=run(root,spec,root=root,prepared=prepared,scenarios=['zero_transaction_cost','configured'],
               publish=False,research_context={'role':'screen','source':'synthetic_v3_interface_fixture'})
    audited=audit_score_account_run(folder)
    return root,folder,audited

def test_v3_native_archive_has_per_fee_primitive_state_audit(archive):
    root,folder,audit=archive
    for fee in ('zero_transaction_cost','configured'):
        assert audit['scenarios'][fee]['actual_holding_policy']['choices']==2
    r=json.loads((folder/'alpha_account_receipt.json').read_text())
    assert r['schema']=='score-account-input-receipt-v3'
    assert not (root/'latest.json').exists()

@pytest.mark.parametrize('mutation,reason',[
    ('shares','Reconstructed actual holding shares'),('last_buy','Reconstructed actual holding last_buy'),
    ('cash','Reconstructed signal-close cash'),('choice','Actual holding targets'),
    ('exit','Actual holding exits'),('scenario','Holding choice schedule/scenario')])
def test_v3_rejects_resigned_state_and_selection_changes(mutation,reason,archive,tmp_path):
    root,original,_=archive;local=tmp_path/'r';shutil.copytree(root/'snapshots',local/'snapshots')
    folder=local/'runs'/original.name;shutil.copytree(original,folder)
    f=folder/'configured/holding_choices.parquet';frame=pd.read_parquet(f)
    state=json.loads(frame.state_json.iloc[1]);selection=json.loads(frame.selection_json.iloc[1])
    if mutation=='shares':state['holdings'][0]['shares']+=100
    elif mutation=='last_buy':state['holdings'][0]['last_buy']='2024-01-11'
    elif mutation=='cash':state['cash']+=1.
    elif mutation=='choice':selection['targets'][0]['reason']='new_entry' if selection['targets'][0]['reason']!='new_entry' else 'retain_actual_holding'
    elif mutation=='exit':selection['exits'][0]['reason']='no_eligible_score'
    else:frame.loc[1,'scenario']='zero_transaction_cost'
    from src.alpharesearch.holding_policy import identity
    selection['state_id']=identity(state)
    frame.loc[1,'state_json']=json.dumps(state);frame.loc[1,'selection_json']=json.dumps(selection)
    frame.to_parquet(f,index=False);resign_archive(folder)
    with pytest.raises(ValueError,match=reason):audit_score_account_run(folder)

@pytest.mark.parametrize('version',['registered-alpha-account-v1','registered-alpha-account-v2'])
def test_legacy_protocol_rejects_hidden_holding_rule(version,tmp_path):
    data,snapshot,spec,hp,_=prepared_case(tmp_path)
    source=dict(kind='registered_learning',experiment_id='exp-'+'a'*12)
    if version.endswith('v2'):source['verification_mode']='reuse_completed_identity_v1'
    doc=dict(schema=version,portfolio=data[3].to_dict(),account=spec.to_dict(),
        scenarios=['zero_transaction_cost','configured'],snapshot_id=snapshot.name,snapshot_manifest_hash=digest(snapshot/'manifest.json'),
        score_source=source,budget=dict(max_input_bytes=1000000,max_task_account_intents=2,max_real_account_intents=1,max_synthetic_account_intents=1))
    protocol(doc)
    doc['holding_policy']=hp.to_dict()
    with pytest.raises(ValueError,match='Strict scored-account protocol'):protocol(doc)

def test_v3_requires_completed_reuse_mode_and_matching_policy(tmp_path):
    data,snapshot,spec,hp,_=prepared_case(tmp_path)
    doc=dict(schema='registered-alpha-account-v3',portfolio=data[3].to_dict(),account=spec.to_dict(),
        holding_policy=hp.to_dict(),scenarios=['zero_transaction_cost','configured'],snapshot_id=snapshot.name,
        snapshot_manifest_hash=digest(snapshot/'manifest.json'),score_source=dict(kind='registered_learning',experiment_id='exp-'+'a'*12,verification_mode='reuse_completed_identity_v1'),
        budget=dict(max_input_bytes=1000000,max_task_account_intents=2,max_real_account_intents=1,max_synthetic_account_intents=1))
    assert protocol(doc)[2]['holding_policy']==hp.to_dict()
    original_policy=doc['holding_policy']
    doc['holding_policy']=ActualHoldingPolicy(3,3,4,.98).to_dict()
    with pytest.raises(ValueError,match='Dynamic policy differs from declared portfolio limits'):protocol(doc)
    doc['holding_policy']=original_policy
    doc['score_source']['verification_mode']='trust_scores'
    with pytest.raises(ValueError,match='Registered learning identity'):protocol(doc)


@pytest.mark.parametrize('mutation',['retain_rank','receipt_version','policy_id'])
def test_native_policy_is_bound_to_registered_definition(mutation,archive,tmp_path):
    from src.researchops.account_output_audit import verify_registered_holding_policy
    _,folder,_=archive
    policy=json.loads((folder/'alpha_holding_policy.json').read_text())
    cfg=dict(spec=dict(schema='registered-alpha-account-v3',holding_policy=policy))
    verify_registered_holding_policy(cfg,folder) # Legal positive before mutation.
    local=tmp_path/'policy';local.mkdir()
    for n in ('alpha_holding_policy.json','alpha_account_receipt.json'):shutil.copyfile(folder/n,local/n)
    if mutation=='retain_rank':cfg['spec']['holding_policy']=dict(policy,retain_rank=4)
    else:
        receipt=json.loads((local/'alpha_account_receipt.json').read_text())
        if mutation=='receipt_version':receipt['schema']='score-account-input-receipt-v2'
        else:receipt['holding_policy_id']='0'*24
        write_json(local/'alpha_account_receipt.json',receipt)
    with pytest.raises(ValueError,match='Native actual holding policy differs from registered v3 policy'):
        verify_registered_holding_policy(cfg,local)
