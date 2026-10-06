"""Pre-registered comparisons of allocation mechanisms, with annual past-only fits."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd

from .artifacts import exclusive_run, content_id, digest, write_json, records
from .contracts import ResearchSpec, Strategy, Execution, Experiment, describe
from .data import prepare_snapshot
from .runner import run, code_inventory
from .regimes import POLICIES, prepare_components, prepare_states, allocation_weights, combine_decisions
from .diagnostics import period_stats
from .foundation_review import joint_block_intervals


def plan():
    base=ResearchSpec(Strategy(name='状态研究',family='allocation',top_k=100,min_history=252,require_fundamentals=True,
        share_basis='known_bonus'),Execution(),Experiment('2022-01-01','2026-09-24'))
    trials=[]
    for policy,(name,rule) in POLICIES.items():
        spec=replace(base,strategy=replace(base.strategy,name=name,allocation_policy=policy))
        trials.append(dict(key=policy,name=name,rule=rule,spec=spec.to_dict(),definition=describe(spec),
                           role='control' if policy in {'small','small_half','defensive','fixed_mix'} else 'candidate'))
    for policy in ['small','hmm2_mix']:
        spec=replace(base,strategy=replace(base.strategy,name=POLICIES[policy][0]+' · 周度',allocation_policy=policy,rebalance='weekly'))
        trials.append(dict(key=policy+'_weekly',name=spec.strategy.name,rule=POLICIES[policy][1],spec=spec.to_dict(),definition=describe(spec),
                           role='control' if policy=='small' else 'candidate'))
    return dict(version=1,kind='regime_allocation',trials=trials,
        question='潜在市场状态能否改善小市值的实际账户，并优于固定组合和简单减仓？',
        evaluation='2022-01-01至2026-09-24；同一起始现金的新账户。不能和2020年起的累计收益直接比较。',
        fitting='每年只用前一年末之前最近756个有效市场日（至少252日）；标准化与参数选择只在训练段。预测使用forward filter，禁止全样本平滑或Viterbi回填。',
        predeclared='2/3状态；4个固定特征；种子11/29仅按训练似然比较；月度主实验、周度配对敏感性；不搜索最优收益参数。',
        decision='先检查数据/因果/账本；分别报告收益、回撤、波动和成本。收益不低且回撤更小才称历史Pareto改善。牺牲收益换风险降低须明确标为取舍。动态模型还需对照固定混合/半仓及延迟信号，不能仅胜过单策略就认定状态有价值。',
        limits='研究者已经看过历史且选择了小市值/低波动组件。本实验滚动计算无未来参数，不等于真正未见样本外。季度估算股本、供应商历史修订、分红应收不复投及日线成交限制继续存在。',
        sources=['https://hmmlearn.readthedocs.io/en/stable/tutorial.html','https://www.nber.org/papers/w22208','https://www.aqr.com/insights/perspectives/factor-timing-is-hard'])


def review(root, rows, states):
    paths={};returns={}
    for r in rows:
        eq=pd.read_parquet(root/'runs'/r['run_id']/'configured/equity.parquet').set_index('date').sort_index()
        paths[r['key']]=eq;returns[r['key']]=eq.equity.div(eq.equity.shift().fillna(1e6)).sub(1)
    matrix=pd.DataFrame(returns)
    if matrix.isna().any().any():raise ValueError('账户日期不一致')
    baseline=next(r for r in rows if r['key']=='small')['metrics']
    controls={'hmm2_mix':'fixed_mix','hmm3_mix':'fixed_mix','hmm2_lag5':'fixed_mix','observable':'fixed_mix',
              'hmm2_cash':'small_half','vol_budget':'small_half','hmm2_mix_weekly':'small_weekly'}
    candidates=[r['key'] for r in rows if r['role']=='candidate']
    active=pd.DataFrame({k:matrix[k]-matrix[controls[k]] for k in candidates})
    bands=joint_block_intervals(active.to_numpy())
    lagged=states.set_index('date').hmm2_p1.reindex(matrix.index).shift(1)
    for row in rows:
        key=row['key']; r=matrix[key];eq=paths[key].reset_index();m=row['metrics']
        annual=[dict(year=y,**period_stats(eq,1e6,g.date.iloc[0],g.date.iloc[-1])) for y,g in eq.groupby(eq.date.str[:4])]
        states_summary=[]
        for label,mask in [('前日高压力概率≥70%',lagged.ge(.7)),('前日高压力概率≤30%',lagged.le(.3)),('前日状态不确定',lagged.gt(.3)&lagged.lt(.7))]:
            sample=r.loc[mask]
            states_summary.append(dict(label=label,sessions=len(sample),mean_daily_return=float(sample.mean()) if len(sample) else None,
                                      volatility=float(sample.std()*np.sqrt(252)) if len(sample)>1 else None))
        evidence=dict(annual=annual,annualized_volatility=float(r.std()*np.sqrt(252)),
            mean_std_ratio_zero_rate=float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else None,
            mean_annual_active_vs_small=float((r-matrix.small).mean()*252),
            states=states_summary,return_difference_vs_small=m['total_return']-baseline['total_return'],
            drawdown_difference_vs_small=m['max_drawdown']-baseline['max_drawdown'])
        if row['role']=='control':verdict='研究对照'
        elif m['total_return']>=baseline['total_return'] and m['max_drawdown']>=baseline['max_drawdown']:verdict='历史收益与回撤同时改善，待复核'
        elif m['max_drawdown']>baseline['max_drawdown']:verdict='降低回撤，但牺牲收益'
        elif m['total_return']>baseline['total_return']:verdict='提高收益，但承担更大回撤'
        else:verdict='当前实现未改善原型'
        if key in candidates:
            j=candidates.index(key);ref=controls[key]
            evidence.update(comparator=ref,mean_annual_active_vs_control=float(active[key].mean()*252),
                joint_active_bands=[dict(block_sessions=b['block_sessions'],lower=b['lower'][j],upper=b['upper'][j]) for b in bands])
        evidence['verdict']=verdict
        row['evidence']=evidence
    return dict(comparison='年度与全期均从同一账户路径计算；均值/标准差指标假定无风险利率为0。联合区间是对配对对照的平均日收益差×252，非CAGR差。',
        controls=controls,candidate_order=candidates,bands=bands,
        trajectories=[dict(key=k,rows=[dict(date=d,nav=float(v/1e6)) for d,v in paths[k].equity.items()]) for k in paths],
        state_daily=records(states),causality='状态图显示当日收盘后可得的过滤概率；状态分组收益使用前一日概率。实际订单使用月/周决策日概率，下一交易日执行。')


def run_suite(project, root=None, progress=lambda m:None):
    project=Path(project);root=Path(root or project/'data/technical');protocol=plan()
    with exclusive_run(root):
        sid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
        folder=root/'regimes'/sid;folder.mkdir(parents=True)
        started=time.perf_counter();protocol.update(suite_id=sid,code_hash=content_id(code_inventory()))
        write_json(folder/'protocol.json',protocol);write_json(folder/'status.json',dict(suite_id=sid,status='running'))
        try:
            base=ResearchSpec.from_dict(protocol['trials'][0]['spec'])
            snapshot,sm=prepare_snapshot(project,root,base,progress)
            progress('生成共同小市值/低波动分支与市场状态特征')
            small,defensive,schedule,market,mf=prepare_components(snapshot,base)
            progress('按年度只使用过去训练，计算2/3状态逐日过滤概率')
            states,models=prepare_states(mf,base.experiment.start_date)
            states.to_parquet(folder/'states.parquet',index=False);write_json(folder/'models.json',models)
            mf.to_parquet(folder/'market_features.parquet',index=False)
            components={'monthly':(small,defensive,schedule,market)};rows=[]
            with ThreadPoolExecutor(max_workers=2) as pool:
                for offset in range(0,len(protocol['trials']),2):
                    jobs={}
                    for trial in protocol['trials'][offset:offset+2]:
                        spec=ResearchSpec.from_dict(trial['spec']);freq=spec.strategy.rebalance
                        if freq not in components:
                            ws,wd,sc,wm,_=prepare_components(snapshot,spec);components[freq]=(ws,wd,sc,wm)
                        ss,ds,sc,mk=components[freq]
                        weights=allocation_weights(states,spec.strategy.allocation_policy,spec.strategy.state_breadth_threshold)
                        table=combine_decisions(ss,ds,weights,spec.strategy.allocation)
                        prepared=dict(snapshot=snapshot,manifest=sm,decisions=table,schedule=sc,market=mk,
                                      extras={'state_probabilities.parquet':weights,'state_models.json':models,
                                              'market_features.parquet':mf,'component_small.parquet':ss,'component_defensive.parquet':ds})
                        jobs[pool.submit(run,project,spec,root=root,prepared=prepared,scenarios=['zero_transaction_cost','configured'],publish=False,
                            research_context=dict(role='screen',suite_id=sid,key=trial['key'],source='regime_allocation',hypothesis=trial['rule']))]=trial
                    for future in as_completed(jobs):
                        trial=jobs[future];run_folder=future.result();r=json.loads((run_folder/'result.json').read_text(encoding='utf-8'))
                        metrics={p['metrics']['scenario']:p['metrics'] for p in r['portfolios']}
                        weights=pd.read_parquet(run_folder/'state_probabilities.parquet')
                        schedule=json.loads((run_folder/'schedule.json').read_text(encoding='utf-8'))
                        alloc=weights.loc[weights.date.isin(schedule),['date','small_fraction','defensive_fraction','cash_fraction']].copy()
                        alloc['execution_date']=alloc.date.map(schedule)
                        rows.append(dict(**trial,run_id=run_folder.name,metrics=metrics['configured'],zero_cost=metrics['zero_transaction_cost'],
                                         execution=r['analysis']['execution'],allocation_decisions=records(alloc)))
                        progress(f"完成 {len(rows)}/{len(protocol['trials'])} {trial['name']}：收益 {metrics['configured']['total_return']:.2%} / 回撤 {metrics['configured']['max_drawdown']:.2%}")
                        write_json(folder/'progress.json',dict(suite_id=sid,rows=rows))
            order={r['key']:i for i,r in enumerate(protocol['trials'])};rows.sort(key=lambda r:order[r['key']])
            evidence=review(root,rows,states)
            if content_id(code_inventory())!=protocol['code_hash']:raise ValueError('运行期间源码发生变化')
            payload=dict(suite_id=sid,status='completed',protocol=protocol,snapshot_id=sm['snapshot_id'],rows=rows,evidence=evidence,models=models,elapsed_seconds=time.perf_counter()-started)
            write_json(folder/'result.json',payload)
            write_json(folder/'manifest.json',dict(suite_id=sid,artifacts={p.name:digest(p) for p in folder.iterdir() if p.is_file() and p.name!='status.json'},
                run_manifests={r['run_id']:digest(root/'runs'/r['run_id']/'manifest.json') for r in rows}))
            write_json(folder/'status.json',dict(suite_id=sid,status='completed'));write_json(root/'latest_regime.json',dict(suite_id=sid))
            return folder
        except Exception as exc:
            write_json(folder/'status.json',dict(suite_id=sid,status='failed',error=str(exc)));raise
