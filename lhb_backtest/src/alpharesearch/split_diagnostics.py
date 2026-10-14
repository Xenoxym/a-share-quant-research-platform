"""Honest saved-score diagnostics; never fit, select features or simulate trades."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .learning import KEYS, CLOCKS, LABELS, _day, _fingerprint
from .contracts import timestamp, timestamps
from ..technical.artifacts import content_id


def diagnose_splits(scores, labels, segments, *, fit_cutoff, as_of, max_rows, score_target):
    """Evaluate declared, nonoverlapping date ranges, preserving unknown labels.

    Training replay is descriptive and never a causal trading backtest. Post-fit
    diagnostics remain retrospective unless separately sealed before execution.
    No validation/test split is invented for experiments that did not use one.
    """
    if type(max_rows) is not int or max_rows<1 or max(len(scores),len(labels))>max_rows:
        raise ValueError('Finite diagnostic row budget exceeded')
    if (not set(KEYS+CLOCKS+['score'])<=set(scores) or not set(KEYS+LABELS)<=set(labels)
            or scores.empty or scores[KEYS].isna().any().any() or labels[KEYS].isna().any().any()
            or scores.duplicated(KEYS).any() or labels.duplicated(KEYS).any()
            or scores.sample_id.duplicated().any() or labels.sample_id.duplicated().any()
            or scores.duplicated(['trade_date','stock_code']).any() or labels.duplicated(['trade_date','stock_code']).any()):
        raise ValueError('Unique complete score and label keys required')
    if (not isinstance(score_target,dict) or set(score_target)!={'kind','unit','predicted_score_is_raw_return'}
            or score_target['kind'] not in {'raw_return','date_centered','date_zscore','date_rank'}
            or score_target['unit']!=('decimal_return' if score_target['kind'] in {'raw_return','date_centered'} else 'dimensionless')
            or type(score_target['predicted_score_is_raw_return']) is not bool
            or score_target['predicted_score_is_raw_return']!=(score_target['kind']=='raw_return')):
        raise ValueError('Explicit consistent score target/units required')
    raw=score_target['predicted_score_is_raw_return']
    cutoff=timestamp(fit_cutoff);now=timestamp(as_of)
    if cutoff>now:raise ValueError('Fit cutoff exceeds evaluation time')
    if not isinstance(segments,list) or not segments:raise ValueError('Explicit diagnostic segments required')
    prior=None;names=set();dates=[]
    for part in segments:
        if (not isinstance(part,dict) or set(part)!={'name','start','end','role','used_for_selection'}
                or part['name'] not in {'train','validation','test'} or part['name'] in names
                or part['role'] not in {'in_sample_replay','post_fit_retrospective'}
                or type(part['used_for_selection']) is not bool):raise ValueError('Strict split declaration required')
        _day(part['start']);_day(part['end'])
        if part['start']>part['end'] or (prior is not None and part['start']<=prior):
            raise ValueError('Chronological nonoverlapping segments required')
        if (part['name']=='train')!=(part['role']=='in_sample_replay'):
            raise ValueError('Training must be labelled in-sample replay')
        prior=part['end'];names.add(part['name']);dates.append(part)
    frame=scores[KEYS+CLOCKS+['score']].copy()
    for key in CLOCKS:frame[key]=timestamps(frame[key])
    if (frame.execution_at.le(frame.decision_at)|frame.target_end_at.le(frame.execution_at)).any():
        raise ValueError('Invalid score clocks')
    if not frame.decision_at.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').eq(frame.trade_date).all():
        raise ValueError('Decision clock differs from local trade date')
    if not np.isfinite(frame.score.to_numpy(dtype=float)).all():raise ValueError('Finite scores required')
    frame=frame.merge(labels[KEYS+LABELS],on=KEYS,how='left',validate='one_to_one',indicator=True)
    if not frame._merge.eq('both').all():raise ValueError('Every scored row requires a label status, including unobserved')
    if not frame.label_observed.map(lambda x:isinstance(x,(bool,np.bool_))).all():
        raise ValueError('Explicit boolean label observation status required')
    for key in ['label_start','label_end','label_known_at']:frame[key]=pd.to_datetime(frame[key].map(lambda x:pd.NaT if pd.isna(x) else timestamp(x)),utc=True)
    observed=frame.label_observed
    if (frame.loc[observed,['target_return','label_start','label_end','label_known_at']].isna().any().any()
            or not np.isfinite(frame.loc[observed,'target_return'].to_numpy(dtype=float)).all()
            or frame.loc[observed,'label_start'].ne(frame.loc[observed,'execution_at']).any()
            or frame.loc[observed,'label_end'].ne(frame.loc[observed,'target_end_at']).any()
            or frame.loc[observed,'label_known_at'].lt(frame.loc[observed,'label_end']).any()):
        raise ValueError('Observed labels must be finite and match score horizons')
    reports=[];covered=pd.Series(False,index=frame.index)
    for part in dates:
        mask=frame.trade_date.between(part['start'],part['end']);covered|=mask;rows=frame.loc[mask]
        if rows.empty:raise ValueError('Declared segment has no scores: '+part['name'])
        if part['role']=='in_sample_replay':
            if rows.decision_at.gt(cutoff).any():raise ValueError('Training replay contains post-fit decisions')
            if (rows.label_observed & (rows.label_known_at.isna()|rows.label_known_at.gt(cutoff))).any():
                raise ValueError('Training labels were not mature at fit cutoff')
        elif rows.decision_at.le(cutoff).any():raise ValueError('Post-fit segment includes pre-fit decisions')
        mature=rows.label_observed & rows.label_known_at.le(now) & rows.target_end_at.le(now)
        used=rows.loc[mature];ics=[]
        for _,day in used.groupby('trade_date',sort=True):
            if len(day)>=3 and day.score.nunique()>1 and day.target_return.nunique()>1:
                ics.append(float(day.score.rank().corr(day.target_return.rank())))
        metrics=None if used.empty else dict(mse=float(mean_squared_error(used.target_return,used.score)) if raw else None,
            mae=float(mean_absolute_error(used.target_return,used.score)) if raw else None,rank_ic_mean=None if not ics else float(np.mean(ics)),rank_ic_dates=len(ics))
        reports.append(dict(**part,scored_rows=len(rows),mature_rows=len(used),unobserved_rows=int((~rows.label_observed).sum()),
            pending_rows=int((rows.label_observed & ~mature).sum()),metrics=metrics,
            error_metric_scope='raw-return score vs mature raw return' if raw else 'MSE/MAE omitted: score target differs from raw return; no evaluation-label transformation',
            comparison_membership=_fingerprint(rows[KEYS+CLOCKS].reset_index(drop=True)),
            mature_labels=_fingerprint(used[KEYS+LABELS].reset_index(drop=True)),
            tradable_history=False,unseen_claim=False))
    if not covered.all():raise ValueError('Every scored row must belong to a declared split')
    result=dict(schema='saved-score-split-diagnostics-v1',fit_cutoff=cutoff.isoformat(),as_of=now.isoformat(),
        score_target=score_target,segments=reports,missing_splits=sorted({'train','validation','test'}-names),model_fits=0,predict_calls=0,account_backtests=0,
        scope='Prediction errors and within-date ordering only; no net return, drawdown or causal training-period account claim')
    result['diagnostics_id']=content_id(result)
    return result
