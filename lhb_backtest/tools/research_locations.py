"""Inspect or explicitly authorize a research-store relocation without editing its database.

Examples: python tools/research_locations.py --root PATH check
          python tools/research_locations.py --root PATH authorize --previous-root OLD_PATH
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.researchops.locations import REGISTRY_NAME, previous_roots, verify_evidence_paths
from src.researchops.service import Research
from src.technical.artifacts import content_id, digest, verify_artifacts, write_json


def authorize(research, previous):
    previous = Path(previous).absolute()
    if '..' in previous.parts or previous == research.root:
        raise ValueError('Previous root must be a different absolute location without traversal')
    roots = previous_roots(research.root)
    if previous in roots:
        return {'already_registered': True}
    affected = 0
    evidence = []
    for task in research.store.list():
        evidence.extend(research.store.get(task['id'])['evidence'])
    database_hash = digest(research.store.path)
    for item in evidence:
        fingerprint, payload = item['fingerprint'], item['payload']
        if content_id(payload['artifacts']) != fingerprint:
            raise ValueError('Evidence content identity differs')
        folder = research.root / 'analysis_evidence' / fingerprint
        if folder.resolve() != folder:
            raise ValueError('Evidence folder is linked')
        verify_evidence_paths(folder, payload['artifacts'])
        verify_artifacts(folder, payload)
        relative = Path('analysis_evidence') / fingerprint
        recorded = Path(payload['folder'])
        if recorded == previous / relative:
            affected += 1
        elif recorded != folder and not any(recorded == root / relative for root in roots):
            raise ValueError('Another unregistered historical location exists')
    if not affected:
        raise ValueError('No evidence belongs to the requested previous root')
    write_json(research.root / REGISTRY_NAME, dict(schema_version=1,
        current_root=str(research.root), previous_roots=[str(p) for p in roots + [previous]],
        authorized_at_utc=datetime.now(timezone.utc).isoformat(),
        reason='Explicit research_locations authorize command; evidence content verified'))
    return {'affected_evidence': affected, 'database_unchanged': digest(research.store.path) == database_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--root')
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('check')
    command = commands.add_parser('authorize')
    command.add_argument('--previous-root', required=True)
    args = parser.parse_args()
    research = Research(args.project, args.root, readonly=True)
    result = authorize(research, args.previous_root) if args.action == 'authorize' else {}
    tasks = 0
    for task in research.store.list():
        if research.store.get(task['id'])['evidence']:
            research.verify_evidence(task['id'])
            tasks += 1
    print(json.dumps(dict(**result, evidence_tasks_checked=tasks, verified=True)))


if __name__ == '__main__':
    main()
