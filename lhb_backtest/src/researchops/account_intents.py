"""Transactional account-intent accounting, including ordinary native controls."""
ALPHA_ZERO = {"alpha_batch", "alpha_screen", "alpha_learn"}
FIELDS = {"max_task_account_intents", "max_real_account_intents", "max_synthetic_account_intents"}

def declared(proposal):
    b=proposal.get("spec",{}).get("budget",{})
    if not FIELDS <= set(b): raise ValueError("Strict account intent budgets required")
    result={k:b[k] for k in FIELDS}
    if any(type(v) is not int or not 1<=v<=1000 for v in result.values()):
        raise ValueError("Bounded integer account intent budgets required")
    if result["max_real_account_intents"]+result["max_synthetic_account_intents"] != result["max_task_account_intents"]:
        raise ValueError("Account sub-budgets must sum to total")
    return result

def count(proposal):
    kind=proposal.get("kind")
    if kind in ALPHA_ZERO: return 0,0
    scenarios=proposal.get("scenarios",proposal.get("spec",{}).get("scenarios"))
    if (not isinstance(scenarios,list) or not scenarios or len(scenarios)!=len(set(scenarios))
            or not set(scenarios)<={"zero_transaction_cost","zero_slippage","configured"}):
        raise ValueError("Explicit account scenarios required in an account-budget task")
    if kind=="alpha_account":
        if type(proposal.get("planned_accounts")) is not int or proposal["planned_accounts"]!=len(scenarios):
            raise ValueError("Strict persisted account count required")
        scope=proposal.get("spec",{}).get("score_source",{}).get("kind")
        if scope not in {"registered_learning","synthetic_stub"}: raise ValueError("Explicit score source kind required")
        return (0,len(scenarios)) if scope=="synthetic_stub" else (len(scenarios),0)
    if kind is None: return len(scenarios),0
    raise ValueError("Unsupported mixed executor in account-budget task")

def enforce_account_budget(proposal,histories):
    accounts=[p for p in histories if p.get("kind")=="alpha_account"]
    if proposal.get("kind")=="alpha_account":
        budget=declared(proposal)
        if any(declared(p)!=budget for p in accounts): raise ValueError("Task account budget already frozen; cannot reset")
    elif accounts: budget=declared(accounts[0])
    else: return
    real,synthetic=0,0
    for p in [*histories,proposal]:
        r,s=count(p);real+=r;synthetic+=s
    if (real>budget["max_real_account_intents"] or synthetic>budget["max_synthetic_account_intents"]
            or real+synthetic>budget["max_task_account_intents"]):
        raise ValueError("Persisted account intent budget exhausted; failures/cancel/freeze count")
