"""Snapshot-bound code rollback. No model calls, reformatting or retrieval changes.

Preview: python -m src.utils.rollback_generation_targets --include-ambiguous
Apply:   python -m src.utils.rollback_generation_targets --include-ambiguous --apply
Run only while writers of the target files are stopped. Backups and reports are
kept even on failure. A second application fails closed on the changed snapshot.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = 'analysis/generation_target_comment_audit_2026-09-18'
TARGETS = {name: f'data/query/black/target/{name}/generation_attack/poison_targets_set_v4.json'
           for name in ('gte', 'harrier')}
SOURCES = {
    'coconut': ('src/preprocessing/bfp/CoCoNut/dataset/c/processed/CoCoNut.jsonl',
                'data/raw/bfp/CoCoNut.jsonl'),
    'codeflaws': ('src/preprocessing/bfp/Codeflaws/dataset/codeflaws.json',
                  'data/raw/bfp/Codeflaws.json'),
    'deepfix': ('src/preprocessing/bfp/Deepfix/dataset/deepfix.json',
                'data/raw/bfp/DeepFix.json'),
}


class RollbackError(ValueError):
    """Unsafe input, snapshot drift, or a failed transaction."""


def require(condition, message):
    if not condition:
        raise RollbackError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def text_hash(value: str) -> str:
    require(isinstance(value, str), 'code must be a string')
    return digest(value.encode('utf-8'))


def object_hash(value) -> str:
    # Matches the audit's published serialization contract.
    return text_hash(json.dumps(value, sort_keys=True, ensure_ascii=False))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def under_root(root: Path, value: str) -> Path:
    """Rebase archived Windows paths onto the selected project root."""
    value = str(value).replace('\\', '/')
    if '/EvilPatch/' in value:
        value = value.rsplit('/EvilPatch/', 1)[1]
    p = (root / value).resolve()
    require(p.is_relative_to(root.resolve()), f'path outside project: {value}')
    return p


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def source_rows(path):
    """One-based physical JSONL line or JSON-array element index."""
    if path.suffix.lower() == '.jsonl':
        with path.open(encoding='utf-8-sig') as handle:
            for number, line in enumerate(handle, 1):
                require(bool(line.strip()), f'blank source line: {path}:{number}')
                yield number, json.loads(line)
    else:
        data = read_json(path)
        require(isinstance(data, list), f'expected JSON array: {path}')
        yield from enumerate(data, 1)


def typed_id(value):
    require(type(value) in (int, str), 'ID must be an integer or string, not bool/null')
    return type(value).__name__, value


@dataclass
class PreparedFile:
    name: str
    path: Path
    before: bytes
    after: bytes
    records: list


@dataclass
class PreparedRun:
    files: list[PreparedFile]
    report: dict
    guard_hashes: dict[Path, str]


def prepare(root=ROOT, manifest=None, review_required=None, include_ambiguous=False):
    """Read-only preflight: verify every selection before preparing any changes."""
    root = Path(root).resolve()
    mp = under_root(root, manifest or f'{AUDIT}/repair_manifest.json')
    rp = under_root(root, review_required or f'{AUDIT}/review_required.json')
    guards = {mp: file_hash(mp)}
    main = read_json(mp)
    selected = list(main['targets'])
    require(all(r['status'] == 'confirmed' for r in selected), 'invalid confirmed manifest')
    if include_ambiguous:
        guards[rp] = file_hash(rp)
        review = read_json(rp)
        require(review['files'] == main['files'], 'review/manifest snapshots differ')
        require(review['provenance_source_sha256'] == main['provenance_source_sha256'],
                'review/manifest source snapshots differ')
        selected += [r for r in review['records'] if r['status'] == 'ambiguous']
    require(bool(selected), 'no selected records')
    keys = [(r['retriever'], r['cwe'], typed_id(r['id'])) for r in selected]
    require(len(keys) == len(set(keys)), 'duplicate selection key')
    require(set(r['retriever'] for r in selected) <= set(TARGETS), 'unsupported target file')

    provenance_hashes = {under_root(root, p): h for p, h in main['provenance_source_sha256'].items()}
    original_rows = {}
    for source in {r['dataset_source'].casefold() for r in selected}:
        require(source in SOURCES, f'unsupported dataset: {source}')
        pre, post = [under_root(root, p) for p in SOURCES[source]]
        for p in (pre, post):
            require(p in provenance_hashes, f'missing audited source fingerprint: {p}')
            require(file_hash(p) == provenance_hashes[p], f'source snapshot changed: {p}')
            guards[p] = provenance_hashes[p]
        group = [r for r in selected if r['dataset_source'].casefold() == source]
        wanted = set()
        for r in group:
            require(len(r['source_matches']) == 1, f'ambiguous source mapping: {r["id"]}')
            match = r['source_matches'][0]
            require(under_root(root, match['pre_format_file']) == pre and
                    under_root(root, match['post_format_file']) == post, 'source path mismatch')
            n = match['row_1based']
            require(type(n) is int and n > 0, 'invalid source row')
            wanted.add(n)
        before_rows = {n: e for n, e in source_rows(pre) if n in wanted}
        after_rows = {n: e for n, e in source_rows(post) if n in wanted}
        require(set(before_rows) == set(after_rows) == wanted, 'source row missing')
        for r in group:
            match = r['source_matches'][0]
            n = match['row_1based']; original = before_rows[n]; formatted = after_rows[n]
            require(typed_id(original['id']) == typed_id(formatted['id']) ==
                    typed_id(match['original_source_id']), f'source ID mismatch at row {n}')
            require(text_hash(formatted['buggy_code']) == r['buggy_sha256'] and
                    text_hash(formatted['fixed_code']) == r['fixed_sha256'],
                    f'formatted pair mismatch at row {n}')
            require(text_hash(original['fixed_code']) == match['pre_format_fixed_sha256'],
                    f'original fixed code mismatch at row {n}')
            text_hash(original['buggy_code'])
            original_rows[(source, n)] = original

    prepared = []
    for name in TARGETS:
        group = [r for r in selected if r['retriever'] == name]
        if not group:
            continue
        path = under_root(root, TARGETS[name]); meta = main['files'][name]
        require(under_root(root, meta['path']) == path, 'unexpected target path')
        raw = path.read_bytes()
        require(digest(raw) == meta['sha256'], f'target snapshot changed; refusing to clear: {path}')
        guards[path] = meta['sha256']
        data = json.loads(raw)
        require(isinstance(data, dict), 'target root must be a CWE mapping')
        index = {}
        for cwe, entries in data.items():
            require(isinstance(entries, list), 'CWE group must be a list')
            for e in entries:
                k = (cwe, typed_id(e['entity']['id']))
                require(k not in index, f'duplicate target key: {name}/{k}')
                index[k] = e
        updated = copy.deepcopy(data)
        updated_index = {(c, typed_id(e['entity']['id'])): e for c, es in updated.items() for e in es}
        changes = []
        selected_keys = set()
        for r in group:
            k = (r['cwe'], typed_id(r['id'])); selected_keys.add(k)
            require(k in index, f'target missing: {name}/{k}')
            e = index[k]; ent = e['entity']; gen = e.get('generation_attack')
            require(isinstance(gen, dict), 'missing generation history')
            require(object_hash(e) == r['entry_sha256'] and object_hash(gen) == r['generation_attack_sha256'],
                    f'entry/history changed: {name}/{k}')
            require(ent.get('source') == r['dataset_source'], 'dataset identity mismatch')
            for field, h in [('buggy_code', 'buggy_sha256'), ('fixed_code', 'fixed_sha256')]:
                require(text_hash(ent[field]) == r[h], f'{field} changed: {name}/{k}')
            require(text_hash(gen['vinj_attack']['vul_code']) == r['vul_sha256'], 'vul_code changed')
            match = r['source_matches'][0]
            original = original_rows[(r['dataset_source'].casefold(), match['row_1based'])]
            new = updated_index[k]
            for field in ('buggy_code', 'fixed_code'):
                new['entity'][field] = original[field]
            del new['generation_attack']
            restored = copy.deepcopy(new)
            restored['entity']['buggy_code'] = ent['buggy_code']
            restored['entity']['fixed_code'] = ent['fixed_code']
            restored['generation_attack'] = gen
            require(restored == e, 'unexpected field changed')
            changes.append(dict(retriever=name, cwe=r['cwe'], id=r['id'], review_status=r['status'],
                                source_match=match, before_entry_sha256=object_hash(e),
                                after_entry_sha256=object_hash(new),
                                before_buggy_sha256=r['buggy_sha256'], before_fixed_sha256=r['fixed_sha256'],
                                after_buggy_sha256=text_hash(original['buggy_code']),
                                after_fixed_sha256=text_hash(original['fixed_code']),
                                removed_field='generation_attack', removed_children=list(gen),
                                generation_attack_sha256=r['generation_attack_sha256']))
        for k, e in index.items():
            if k not in selected_keys:
                require(updated_index[k] == e, f'unselected entry changed: {name}/{k}')
        after = (json.dumps(updated, ensure_ascii=False, indent=2) + '\n').encode('utf8')
        require(json.loads(after) == updated, 'serialized output mismatch')
        prepared.append(PreparedFile(name, path, raw, after, changes))
    check_guards(guards)
    report = dict(schema_version=1, status='preflight_passed', include_ambiguous=include_ambiguous,
                  created_at=datetime.now(timezone.utc).isoformat(),
                  warning='Exact pre-format restoration is not semantic comment recovery. No models are invoked.',
                  files=[dict(retriever=f.name, path=str(f.path), before_sha256=digest(f.before),
                              after_sha256=digest(f.after), count=len(f.records),
                              cwe_counts=dict(Counter(r['cwe'] for r in f.records)), records=f.records)
                         for f in prepared],
                  input_fingerprints={str(p): h for p, h in guards.items()})
    return PreparedRun(prepared, report, guards)


def check_guards(guards):
    for p, expected in guards.items():
        require(file_hash(p) == expected, f'file changed during preflight: {p}')


def exclusive_write(path, raw):
    with path.open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())


def stage_file(path, raw):
    fd, temp = tempfile.mkstemp(prefix=f'.{path.name}.rollback-', suffix='.tmp', dir=path.parent)
    temp = Path(temp)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        require(file_hash(temp) == digest(raw), 'staged bytes mismatch')
        return temp
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def save_report(directory, report):
    path = directory / 'execution_report.json'
    raw = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf8')
    temp = stage_file(path, raw)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def apply_run(run, backup_root):
    """All backups precede target writes; replace each file, recover on failure."""
    check_guards(run.guard_hashes)
    backup_root = Path(backup_root)
    for f in run.files:
        require(backup_root.resolve() != f.path, 'backup path aliases input')
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory = backup_root / stamp
    directory.mkdir(exist_ok=False)
    report = copy.deepcopy(run.report)
    report['backup_directory'] = str(directory)
    report['status'] = 'backing_up'
    staged, committed = {}, []
    try:
        save_report(directory, report)
        for f, info in zip(run.files, report['files']):
            backup = directory / f'{f.name}.original.json'
            exclusive_write(backup, f.before)
            require(file_hash(backup) == digest(f.before), 'backup verification failed')
            info['backup_path'] = str(backup)
            staged[f.name] = stage_file(f.path, f.after)
            require(json.loads(staged[f.name].read_bytes()) == json.loads(f.after), 'staged JSON mismatch')
        check_guards(run.guard_hashes)
        report['status'] = 'prepared'
        save_report(directory, report)
        for f in run.files:
            require(file_hash(f.path) == digest(f.before), f'concurrent target modification: {f.path}')
            report['status'] = 'committing'; report['next_file'] = f.name
            save_report(directory, report)
            os.replace(staged[f.name], f.path)
            committed.append(f)
            require(file_hash(f.path) == digest(f.after), 'post-write verification failed')
            report['committed'] = [x.name for x in committed]
            save_report(directory, report)
        for f in run.files:
            require(file_hash(f.path) == digest(f.after), f'target changed after write: {f.path}')
        report['status'] = 'completed'; report.pop('next_file', None)
        save_report(directory, report)
        return directory
    except BaseException as exc:
        report['status'] = 'failed'; report['error'] = f'{type(exc).__name__}: {exc}'
        report['recovery'] = []
        for f in reversed(committed):
            recovery = dict(retriever=f.name)
            temp = None
            try:
                require(file_hash(f.path) == digest(f.after), 'current file differs; automatic restore refused')
                temp = stage_file(f.path, f.before)
                require(file_hash(f.path) == digest(f.after), 'file changed before recovery')
                os.replace(temp, f.path)
                require(file_hash(f.path) == digest(f.before), 'recovery hash mismatch')
                recovery['status'] = 'original_restored'
            except BaseException as recovery_exc:
                recovery.update(status='manual_recovery_required', error=str(recovery_exc))
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
            report['recovery'].append(recovery)
        try:
            save_report(directory, report)
        except OSError:
            pass  # Earlier durable journal and verified backups remain available.
        raise RollbackError(f'Apply failed. Backups/journal: {directory}; {exc}') from exc
    finally:
        for temp in staged.values():
            temp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--manifest', default=f'{AUDIT}/repair_manifest.json')
    parser.add_argument('--review-required', default=f'{AUDIT}/review_required.json')
    parser.add_argument('--include-ambiguous', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup-dir', default='analysis/generation_target_rollback_backups')
    args = parser.parse_args(argv)
    try:
        run = prepare(args.project_root, args.manifest, args.review_required, args.include_ambiguous)
        for f in run.files:
            print(f'{f.name}: {len(f.records)} targets; CWE counts: {dict(Counter(r["cwe"] for r in f.records))}')
        if args.apply:
            directory = apply_run(run, under_root(args.project_root, args.backup_dir))
            print(f'Completed. Verified backups and execution report: {directory}')
        else:
            print('Preflight passed. No files modified. Use --apply to back up and write.')
        return 0
    except (RollbackError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f'ERROR: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
