"""Independent frozen-protocol, publication-time and account audit of a foundation suite."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.technical.artifacts import digest, verify_artifacts, write_json
from src.technical.runner import verify
from verify_technical import audit


def check(folder):
    folder = Path(folder)
    root = folder.parent.parent
    result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    verify_artifacts(folder, manifest)
    protocol = result['protocol']
    assert len(result['rows']) == len(protocol['trials']) == 15
    accounts, snapshots = [], set()
    declared = {v['key']: v for v in protocol['trials']}
    for row in result['rows']:
        run = root/'runs'/row['run_id']
        assert digest(run/'manifest.json') == manifest['run_manifests'][run.name]
        rm, snapshot, sm = verify(run)
        assert rm['spec'] == declared[row['key']]['spec']
        assert rm['code_hash'] == protocol['code_hash']
        snapshots.add(rm['snapshot_id'])
        decisions = pd.read_parquet(run/'decisions.parquet')
        selected = decisions.loc[decisions.selected]
        for prefix in ['q_', 'a_']:
            assert selected[prefix+'publication_date'].lt(selected.trade_date).all()
            assert selected[prefix+'report_date'].le(selected[prefix+'publication_date']).all()
        assert selected.q_age_days.between(0, 240).all()
        assert selected.a_age_days.between(0, 550).all()
        assert selected.execution_date.gt(selected.trade_date).all()
        assert selected.groupby('trade_date').size().le(100).all()
        np.testing.assert_allclose(selected.estimated_market_cap, selected.close*selected.q_total_shares, rtol=1e-12)
        # Fee/cash/share/open-price identities come from independent ledger recomputation.
        a = audit(run)
        a.update(key=row['key'], selected_rows=len(selected))
        accounts.append(a)
    assert snapshots == {result['snapshot_id']}
    return {'status':'pass', 'suite_id':result['suite_id'], 'accounts':accounts,
            'checks':['15 definitions match predeclared protocol and frozen source',
                      'all accounts share one data snapshot',
                      'all selected reports published strictly before decision; no stale admitted reports',
                      'market-cap proxy recomputed from saved raw close and disclosed shares',
                      '30 scenario cash/share/receivable/equity identities and fees independently reconciled'],
            'limits':'These checks verify internal consistency, not vendor truth or original historical financial vintages.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('suite',type=Path); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); value=check(a.suite); write_json(a.output,value)
    print(json.dumps({'status':value['status'],'suite_id':value['suite_id'],'accounts':len(value['accounts'])}))
