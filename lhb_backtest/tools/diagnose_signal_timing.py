"""Forward labels only: not a tradable portfolio, never used for candidate selection."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.technical.artifacts import write_json


def main():
    root=PROJECT/'data/technical'
    batch=json.loads((root/'batches/20260926T092628-634a6ba0/result.json').read_text(encoding='utf-8'))
    snapshot=root/'snapshots'/batch['snapshot_id']
    cal=json.loads((snapshot/'calendar.json').read_text(encoding='utf-8'))
    day_index={d:i for i,d in enumerate(cal)}
    bars=pd.read_parquet(snapshot/'bars.parquet',columns=['stock_code','trade_date','open','close','pre_close'])
    bars=bars.sort_values(['stock_code','trade_date'])
    valid=(bars.close>0)&(bars.pre_close>0)&(bars.open>0)
    bars['log_return']=np.log((bars.close/bars.pre_close).where(valid))
    bars['reference_open']=np.exp(bars.groupby('stock_code').log_return.cumsum())*bars.open/bars.close
    q=bars.set_index(['stock_code','trade_date']).reference_open
    output=[]
    for trial in [9,14,19]:
        row=next(r for r in batch['rows'] if r['trial']==trial)
        p=root/'runs'/row['run_id']
        selected=pd.read_parquet(p/'decisions.parquet').loc[lambda d:d.selected].copy()
        dates=sorted(selected.trade_date.unique())
        next_rebalance=dict(zip(dates[:-1],dates[1:]))
        selected=selected.loc[selected.trade_date.isin(next_rebalance)].copy()
        selected['exit']=selected.trade_date.map(next_rebalance).map(lambda d:cal[day_index[d]+1])
        for delay in [0,1]:
            selected['entry']=selected.execution_date.map(lambda d:cal[day_index[d]+delay])
            start=q.reindex(pd.MultiIndex.from_arrays([selected.stock_code,selected.entry])).to_numpy()
            end=q.reindex(pd.MultiIndex.from_arrays([selected.stock_code,selected.exit])).to_numpy()
            r=end/start-1
            selected['forward_label']=r
            output.append({'trial':trial,'extra_entry_delay_sessions':delay,'rows':len(r),
                'missing_label_rows':int((~np.isfinite(r)).sum()),
                'mean_label':float(np.nanmean(r)),'median_label':float(np.nanmedian(r)),
                'fraction_positive':float(np.nanmean(np.where(np.isfinite(r),r>0,np.nan))),
                'yearly_mean':selected.groupby(selected.trade_date.str[:4]).forward_label.mean().to_dict()})
    payload={'notice':'Forward reference-price labels, not executable returns. Excludes transaction costs, fill constraints and cash. Missing endpoints disclosed; reference price is not official total return. Entry at next open or one extra session later; common exit at next scheduled rebalance open. Descriptive post-screen diagnostic only.','rows':output}
    write_json(PROJECT/'research/research_loop_20260926/signal_timing_labels.json',payload)
    print(json.dumps(payload,ensure_ascii=True))


if __name__=='__main__':main()
