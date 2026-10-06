"""Two bounded post-screen diagnostics, declared before outcomes are inspected."""
import json
from dataclasses import replace
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.technical.artifacts import write_json
from src.technical.contracts import ResearchSpec
from src.technical.runner import run
from src.technical.diagnostics import execution_for, inspect_period


def main():
    root = PROJECT / "data/technical"
    batch = json.loads((root / "batches/20260926T092628-634a6ba0/result.json").read_text(encoding="utf-8"))
    selected = next(r for r in batch["rows"] if r["trial"] == batch["selected_trial"])
    severe = next(r for r in batch["rows"] if r["trial"] == 14)
    base = ResearchSpec.from_dict(json.loads((root / "runs" / selected["run_id"] / "spec.json").read_text(encoding="utf-8")))
    stress = ResearchSpec.from_dict(json.loads((root / "runs" / severe["run_id"] / "spec.json").read_text(encoding="utf-8")))
    experiments = [
        ("turnover_cost_diagnostic", replace(stress, strategy=replace(stress.strategy, name="复核 · 5日反转周调仓 · 三种成本")),
         "严重亏损是否主要由高频换仓费用造成？固定信号，独立重跑三个成本账户。"),
        ("concentration_diagnostic", replace(base, strategy=replace(base.strategy, top_k=100, name="复核 · 10日反转 MA126 · 分散至100只")),
         "将前20只扩展为前100只，检验结果是否依赖极端排名及集中持仓。股票集合与换手会同时改变，不能宣称是纯粹的集中度因果效应。"),
    ]
    folder = PROJECT / "research/research_loop_20260926"
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {"parent_batch": batch["batch_id"], "status": "post_screen_retrospective_diagnostic",
        "selection_notice": "由已看到的失败结果提出；不用于宣称未见样本外或自动挑选盈利参数。",
        "experiments": [{"id": key, "hypothesis": h, "spec": s.to_dict()} for key, s, h in experiments]}
    if (folder / "followup_protocol.json").exists():
        raise ValueError("本轮协议已存在，避免无记录重复尝试")
    write_json(folder / "followup_protocol.json", protocol)
    rows = []
    for key, spec, hypothesis in experiments:
        p = run(PROJECT, spec, root=root, publish=False,
            research_context={"role": "post_screen_diagnostic", "parent_batch": batch["batch_id"], "hypothesis": hypothesis},
            progress=lambda m: print(m, flush=True))
        result = json.loads((p / "result.json").read_text(encoding="utf-8"))
        rows.append({"id": key, "run_id": p.name, "hypothesis": hypothesis,
            "cost_scenarios": [r["metrics"] for r in result["portfolios"]],
            "validation": inspect_period(p, "2024-01-01", result["data_end"])["stats"],
            "execution": execution_for(p)})
        write_json(folder / "followup_results.json", {"protocol": protocol, "rows": rows})
        print(key, p.name, flush=True)


if __name__ == "__main__":
    main()
