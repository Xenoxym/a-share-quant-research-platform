"""New registered MLP integration counterexamples; one tiny fit, no market data."""
import copy,json,hashlib,shutil
from pathlib import Path
import joblib,numpy as np,pandas as pd,pytest
from src.alpharesearch.models import model_spec
from src.alpharesearch.learning import LearningSpec,_prepare,_prepared_parts
from src.alpharesearch.checkpoints import load_checkpoint
from src.researchops.learn_experiments import inspect_inputs
from src.researchops.learn_output_audit import _model_metadata
from src.researchops.screen_experiments import assembly_reference
from src.researchops.resources import WorkerResources
from src.technical.artifacts import digest,write_json
from tests.test_alpha_learn_registration import setup

@pytest.fixture(scope="module")
def registered_mlp(tmp_path_factory):
 root=tmp_path_factory.mktemp("m")
 ops,session,proposal,folder=setup(root,intent_budget=1)
 roles={b['role']:ops.project/b['path'] for b in proposal['inputs']}
 # Five disjoint sets of stocks; remove one mature-date row to exercise unequal day weights.
 for role in ['decisions','labels','assembly_values','assembly_missing']:
  frame=pd.read_parquet(roles[role]);copies=[]
  for group in range(5):
   block=frame.copy();block['stock_code']=[f"{int(v[:6])+10*group:06}.SH" for v in block.stock_code];block['sample_id']=block.trade_date+'-'+block.stock_code;copies.append(block)
  expanded=pd.concat(copies,ignore_index=True);expanded=expanded.loc[~(expanded.trade_date.eq('2024-01-03') & expanded.stock_code.eq('600043.SH'))].reset_index(drop=True)
  assert len(expanded)==99;expanded.to_parquet(roles[role],index=False)
 # Recompute legitimate materialization receipt for new synthetic membership before registration.
 from src.alpharesearch.assembly import FeatureAssembly
 from src.alpharesearch.features.base import FeatureBlock
 from src.alpharesearch.registry import FeatureRegistry
 registry=FeatureRegistry.from_dict(json.loads(roles['registry'].read_text(encoding='utf8')))
 definition=json.loads(roles['assembly_definition'].read_text(encoding='utf8'));values=pd.read_parquet(roles['assembly_values']);missing=pd.read_parquet(roles['assembly_missing']);metadata=dict(definition);units=metadata.pop('units')
 assembly=FeatureAssembly(FeatureBlock(values,missing,units,metadata),registry,tuple(metadata['source_bindings']))
 write_json(roles['assembly_receipt'],assembly_reference(assembly,{r:digest(roles[r]) for r in ['assembly_values','assembly_missing','assembly_definition']}))
 for b in proposal['inputs']:b.update(sha256=digest(roles[b['role']]),bytes=roles[b['role']].stat().st_size)
 cfg=proposal['spec']['learning'];cfg.update(schema='causal-learning-v2',weighting='equal_decision_date',max_input_rows=100,training_target=dict(kind='date_centered',min_date_rows=2,scale_floor=1e-12,unusable_policy='reject'));proposal['spec']['learning']=LearningSpec.from_dict(cfg).to_dict()
 proposal['spec'].update(schema='registered-alpha-learn-v3',model=model_spec('mlp',params=dict(hidden_units=8,max_iter=3,batch_size=32,alpha=1.0,learning_rate_init=0.01),max_train_rows=100,max_features=2,max_matrix_cells=1000).to_dict())
 proposal['spec']['budget']['max_checkpoint_bytes']=1000000;proposal['timeout_seconds']=180
 proposal['resources']=WorkerResources(library_threads=2,max_process_rss_bytes=900000000,min_available_memory_bytes=2*2**30,max_output_bytes=5000000).to_dict()
 spec,model,_,info=inspect_inputs(ops.project,proposal,include_data=True);assert info['prepared'][0].shape==(39,2)
 ex=ops.register(session,proposal);done=ops.execute(session,ex['id']);assert done['status']=='completed',done.get('error')
 ex=ops.store.experiment(ex['id']);assert ex['result']['verified'] is True and ex['result']['model_fits']==1 and ex['result']['additional_audit_fits']==0
 out=ops.root/'worker_jobs'/ex['id']/'learn_result';receipt=json.loads((out/'learning_receipt.json').read_text(encoding='utf8'));assert receipt['actual_fit_attempts_this_call']==1
 write_json(root/'integration_case.json',dict(project=str(ops.project),root=str(ops.root),task_id=session['task_id'],experiment_id=ex['id'],synthetic_registered_fits=1,market_fits=0,accounts=0,rows=39,predictions=40,receipt_sha256=digest(out/'learning_receipt.json'),audit_sha256=digest(ops.root/'worker_jobs'/ex['id']/'alpha_audit.json')))
 return dict(ops=ops,proposal=proposal,out=out,receipt=receipt,model=model,info=info)

def test_registered_mlp_passes_real_frozen_output_audit(registered_mlp):
 c=registered_mlp;r=c['receipt'];assert r['mature_training_rows']==39 and r['prediction_rows']==40 and r['observed_but_immature_labels']==20
 assert r['fitted_model']['scaling']=='train-only mean/std' and r['fitted_model']['neural_training']['iterations']==3 and r['fitted_model']['neural_training']['internal_random_validation'] is False
 tr=json.loads((c['out']/'training_artifact_receipt.json').read_text(encoding='utf8'));assert tr['additional_model_fits']==0 and tr['actual_raw_targets_saved'] and tr['effective_row_weights_saved']

def test_maturity_date_weights_centered_targets_are_actual_values(registered_mlp):
 c=registered_mlp;t=pd.read_parquet(c['out']/'training_trace.parquet');assert len(t)==39 and set(t.trade_date)=={'2024-01-02','2024-01-03'}
 for day,block in t.groupby('trade_date'):
  assert float(block.weight_used.sum())==pytest.approx(39/2);assert float(block.target_return_used.mean())==pytest.approx(0,abs=1e-14)
 assert sorted(t.groupby('trade_date').size())==[19,20]

def test_unavailable_labels_do_not_change_preparation(registered_mlp):
 c=registered_mlp;spec,model,_,info=inspect_inputs(c['ops'].project,c['proposal'],include_data=True);decisions,assembly,labels=info['data'];bad=labels.copy();mask=bad.trade_date.ge('2024-01-04');bad.loc[mask,'target_return']=99999.;bad.loc[mask,'label_known_at']='2099-01-01T00:00:00Z'
 before=_prepared_parts(_prepare(spec,model,decisions,assembly,labels));after=_prepared_parts(_prepare(spec,model,decisions,assembly,bad))
 for a,b in zip(before[:4],after[:4]):
  if a is None:assert b is None
  elif isinstance(a,pd.Series):pd.testing.assert_series_equal(a,b,check_exact=True)
  else:pd.testing.assert_frame_equal(a,b,check_exact=True)

@pytest.mark.parametrize('mutation',['scaling','iterations_bool','loss_nan','unknown_convergence_warning','false_convergence_claim','internal_validation'])
def test_neural_metadata_tampering_rejected_without_fit(registered_mlp,mutation):
 c=registered_mlp;m=copy.deepcopy(c['receipt']['fitted_model']);n=m['neural_training'];X,y,w,*_=c['info']['prepared']
 if mutation=='scaling':m['scaling']='none'
 elif mutation=='iterations_bool':n['iterations']=True
 elif mutation=='loss_nan':n['loss_curve'][0]=float('nan')
 elif mutation=='unknown_convergence_warning':n['recorded_training_warnings'].append(dict(category='ConvergenceWarning',message='unexpected'))
 elif mutation=='false_convergence_claim':n['convergence_reached']=True
 else:n['internal_random_validation']=True
 with pytest.raises(ValueError):_model_metadata(m,c['model'],X,w)

@pytest.mark.parametrize('mutation',['actual_epochs','actual_loss','actual_alpha','actual_validation'])
def test_saved_estimator_tampering_rejected_without_refit(registered_mlp,tmp_path,mutation):
 c=registered_mlp;folder=tmp_path/'cp';shutil.copytree(c['out']/'model_checkpoint',folder);doc=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
 # Caller-generated trusted fixture, never deserialize an external checkpoint.
 model=joblib.load(folder/'model.joblib');est=model.named_steps['regressor']
 if mutation=='actual_epochs':est.n_iter_+=1
 elif mutation=='actual_loss':est.loss_curve_[0]+=0.1
 elif mutation=='actual_alpha':est.alpha+=1.0
 else:est.early_stopping=True
 joblib.dump(model,folder/'model.joblib',compress=0,protocol=5);doc.update(model_sha256=digest(folder/'model.joblib'),model_bytes=(folder/'model.joblib').stat().st_size);write_json(folder/'manifest.json',doc)
 with pytest.raises(ValueError,match='MLP saved estimator'):load_checkpoint(folder,expected_manifest_sha256=digest(folder/'manifest.json'),trusted_local=True,max_bytes=1000000)
