"""Independent daily cash, share, raw-price and fee audit of technical runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.technical.runner import verify
from src.technical.artifacts import write_json


def audit(folder, *, quotes=None):
    folder=Path(folder)
    manifest,snapshot,sm=verify(folder)
    spec=manifest['spec']; ex=spec['execution']
    assert not any('lhb_' in Path(p['path']).name for p in sm['sources'])
    if quotes is None:
        quotes=pd.read_parquet(snapshot/'bars.parquet',columns=['stock_code','trade_date','open']).set_index(['stock_code','trade_date'])
    output={'run_id':folder.name,'no_lhb_inputs':True,'scenarios':{}}
    result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    for scenario in [p['metrics']['scenario'] for p in result['portfolios']]:
        p=folder/scenario
        eq=pd.read_parquet(p/'equity.parquet').set_index('date')
        fills=pd.read_parquet(p/'fills.parquet')
        for key in ['price','filled_shares','commission','other_fee','stamp_tax','slippage','total']:
            fills[key]=fills[key].astype(float)
        ledger=pd.read_parquet(p/'ledger.parquet')
        positions=pd.read_parquet(p/'positions.parquet')
        cash=ledger.groupby('date').cash_flow.sum().reindex(eq.index,fill_value=0).cumsum()+ex['initial_cash']
        receipts=ledger.groupby('date').receivable_flow.sum().reindex(eq.index,fill_value=0).cumsum()
        mv=positions.groupby('date').market_value.sum().reindex(eq.index,fill_value=0)
        errors={'cash':float((cash-eq.cash).abs().max()),'receivable':float((receipts-eq.receivable).abs().max()),
            'equity':float((cash+receipts+mv-eq.equity).abs().max())}
        assert max(errors.values())<.001,errors
        balances=ledger.groupby(['date','stock_code']).shares_delta.sum().unstack(fill_value=0).reindex(eq.index,fill_value=0).cumsum()
        observed=positions.pivot(index='date',columns='stock_code',values='shares').reindex(index=eq.index,columns=balances.columns).fillna(0)
        pd.testing.assert_frame_equal(balances.astype(float),observed.astype(float),check_names=False)
        assert (fills.loc[fills.side.eq('buy'),'filled_shares']%100==0).all()
        keys=pd.MultiIndex.from_arrays([fills.stock_code,fills.date])
        np.testing.assert_allclose(fills.price,quotes.open.reindex(keys),rtol=0,atol=1e-10)
        notion=fills.price*fills.filled_shares
        commission=np.maximum(ex['minimum_commission'],notion*ex['commission_rate'])
        other=notion*ex['other_fee_rate']
        stamp=notion*np.where(fills.date.lt('2023-08-28'),.001,.0005)*fills.side.eq('sell')
        slip=notion*ex['slippage_bps']/10000 if scenario=='configured' else np.zeros(len(fills))
        if scenario=='zero_transaction_cost':
            commission=other=stamp=np.zeros(len(fills))
        for k,expected in [('commission',commission),('other_fee',other),('stamp_tax',stamp),('slippage',slip),('total',commission+other+stamp+slip)]:
            np.testing.assert_allclose(fills[k],expected,rtol=0,atol=1e-8)
        # No same-session repurchase/sale of a stock; all sold shares pre-exist.
        assert not fills.groupby(['date','stock_code']).side.nunique().gt(1).any()
        output['scenarios'][scenario]={'daily_errors':errors,'fills':len(fills),'fees':float(fills.total.sum())}
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit(args.run)
    if args.output:write_json(args.output,result)
    print(json.dumps(result,indent=2))
