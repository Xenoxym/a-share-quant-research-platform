"""Relocation must preserve evidence checks and reject unapproved or escaping locations."""
import json
from pathlib import Path
import shutil

import pytest

from src.researchops.locations import evidence_folder, verify_evidence_paths
from src.researchops.service import Research
from src.technical.artifacts import content_id, digest

TASK = dict(title='Relocation regression', question='Is identical evidence accessible after a move?',
            rationale='Physical location is not content identity', success_criteria=['Hash checks still apply'],
            stop_criteria=['Unapproved location or modified evidence must fail'])


def relocated(tmp_path):
    project = tmp_path / 'app'
    source = project / 'research' / 'proof'
    source.mkdir(parents=True)
    (source / 'report.md').write_text('Stable frozen evidence', encoding='utf-8')
    old = tmp_path / 'old_store'
    r = Research(project, old)
    task = r.create(TASK)
    session = r.store.claim(task['id'], 'fixture')
    evidence = r.attach(session, dict(path='research/proof', kind='feature_analysis', summary='A frozen proof'))
    r.verify_evidence(task['id'])
    new = tmp_path / 'new_store'
    shutil.move(str(old), str(new))
    return project, old, new, task['id'], evidence


def authorize(old, new):
    (new / 'location_aliases.json').write_text(json.dumps(dict(schema_version=1,
        current_root=str(new.resolve()), previous_roots=[str(old.resolve())])), encoding='utf-8')


def test_move_requires_explicit_alias_and_keeps_historical_payload(tmp_path):
    project, old, new, task, evidence = relocated(tmp_path)
    r = Research(project, new, readonly=True)
    with pytest.raises(ValueError, match='authorized relocation'):
        r.verify_evidence(task)
    database_hash = digest(new / 'research.sqlite3')
    authorize(old, new)
    r.verify_evidence(task)
    context = r.context(task)
    assert context['task']['evidence'][0]['payload']['folder'] == evidence['payload']['folder']
    assert context['task']['evidence'][0]['access_folder'] == str(new / 'analysis_evidence' / evidence['fingerprint'])
    assert digest(new / 'research.sqlite3') == database_hash


def test_authorized_move_does_not_accept_modified_file(tmp_path):
    project, old, new, task, evidence = relocated(tmp_path)
    authorize(old, new)
    (new / 'analysis_evidence' / evidence['fingerprint'] / 'report.md').write_text('Changed')
    with pytest.raises(ValueError):
        Research(project, new, readonly=True).verify_evidence(task)


def test_registry_for_another_store_is_rejected(tmp_path):
    project, old, new, task, _ = relocated(tmp_path)
    (new / 'location_aliases.json').write_text(json.dumps(dict(schema_version=1,
        current_root=str(tmp_path / 'wrong'), previous_roots=[str(old)])), encoding='utf-8')
    with pytest.raises(ValueError, match='registry'):
        Research(project, new, readonly=True).verify_evidence(task)


def test_mapping_does_not_accept_other_subtrees(tmp_path):
    root = tmp_path / 'new';root.mkdir()
    old = tmp_path / 'old';authorize(old, root)
    fingerprint = 'a' * 64
    with pytest.raises(ValueError):
        evidence_folder(root, str(old / 'other' / fingerprint), fingerprint)


@pytest.mark.parametrize('fingerprint', ['../../outside', 'a' * 63, 'g' * 64, None])
def test_invalid_fingerprint_cannot_select_a_folder(tmp_path, fingerprint):
    with pytest.raises(ValueError):
        evidence_folder(tmp_path, str(tmp_path), fingerprint)


@pytest.mark.parametrize('name', ['../outside.md', '/outside.md', 'C:/outside.md', 'sub\\outside.md'])
def test_evidence_manifest_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError):
        verify_evidence_paths(tmp_path, {name: 'a' * 64})


def test_parent_traversal_in_recorded_location_is_rejected(tmp_path):
    fingerprint = 'a' * 64
    with pytest.raises(ValueError, match='traversal'):
        evidence_folder(tmp_path, str(tmp_path / 'x' / '..' / 'analysis_evidence' / fingerprint), fingerprint)


def test_symlinked_artifact_cannot_bypass_location_checks(tmp_path):
    target = tmp_path / 'target.md';target.write_text('External')
    folder = tmp_path / 'evidence';folder.mkdir()
    try:
        (folder / 'report.md').symlink_to(target)
    except OSError:
        pytest.skip('Host does not grant symlink creation')
    with pytest.raises(ValueError, match='link'):
        verify_evidence_paths(folder, {'report.md': digest(target)})

def test_operator_authorization_verifies_content_before_writing_registry(tmp_path):
    from tools.research_locations import authorize as register_move
    project, old, new, task, evidence = relocated(tmp_path)
    folder = new / 'analysis_evidence' / evidence['fingerprint']
    (folder / 'report.md').write_text('Wrong content')
    with pytest.raises(ValueError):
        register_move(Research(project, new, readonly=True), old)
    assert not (new / 'location_aliases.json').exists()


def test_operator_authorization_preserves_database_and_is_idempotent(tmp_path):
    from tools.research_locations import authorize as register_move
    project, old, new, task, evidence = relocated(tmp_path)
    r = Research(project, new, readonly=True)
    result = register_move(r, old)
    assert result['affected_evidence'] == 1 and result['database_unchanged']
    r.verify_evidence(task)
    assert register_move(r, old)['already_registered']


def test_operator_does_not_register_an_unrelated_root(tmp_path):
    from tools.research_locations import authorize as register_move
    project, old, new, task, evidence = relocated(tmp_path)
    with pytest.raises(ValueError):
        register_move(Research(project, new, readonly=True), tmp_path / 'unrelated')
    assert not (new / 'location_aliases.json').exists()
