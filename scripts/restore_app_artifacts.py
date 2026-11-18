"""Restore only registry-approved checkpoint bytes; never overwrite existing files.

Sources are explicit checkpoint files or ZIP archives. ZIP members are read in
memory, never extracted by their names. No checkpoint is deserialized here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def restore(root, sources, apply=False):
    root = Path(root).resolve()
    models = json.loads((root / 'configs/real_model_registry.json').read_text(encoding='utf-8'))['models']
    approved = {m['sha256']: m for m in models}
    found = {}
    for source in sources:
        source = Path(source).resolve(strict=True)
        data = source.read_bytes()
        digest = sha256(data)
        if digest in approved:
            found.setdefault(digest, (data, str(source), None))
            continue
        if zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                for member in archive.infolist():
                    if Path(member.filename).suffix.lower() not in {'.pt', '.pth', '.ckpt'}:
                        continue
                    # Guard against accidentally loading unrelated huge archives.
                    if member.file_size > 512 * 1024 * 1024:
                        raise ValueError(f'Checkpoint member too large: {member.filename}')
                    data = archive.read(member)
                    digest = sha256(data)
                    if digest in approved:
                        found.setdefault(digest, (data, str(source), member.filename))
    plan = []
    for model in models:
        target = (root / model['artifact_path']).resolve()
        if not target.is_relative_to(root / 'artifacts'):
            raise ValueError('Artifact destination must remain inside artifacts/')
        expected = model['sha256']
        entry = dict(instance=f"{model['method']}-{model['category']}",
                     version=model['version'], path=model['artifact_path'], sha256=expected)
        if target.exists():
            if not target.is_file() or sha256(target.read_bytes()) != expected:
                raise ValueError(f'Refusing to overwrite existing artifact: {target}')
            entry.update(status='already_verified', bytes=target.stat().st_size)
        elif expected in found:
            data, source, member = found[expected]
            entry.update(status='ready', source=source, member=member, bytes=len(data))
        else:
            raise FileNotFoundError(f'No SHA-256 matching source for {entry["instance"]}')
        plan.append(entry)
    # Preflight every destination and source before creating anything.
    if apply:
        for entry in plan:
            if entry['status'] == 'ready':
                target = root / entry['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as output:
                    output.write(found[entry['sha256']][0])
                if sha256(target.read_bytes()) != entry['sha256']:
                    raise RuntimeError(f'Restored hash mismatch: {target}')
                entry['status'] = 'restored_verified'
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source', type=Path, action='append', default=[])
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = restore(args.root, args.source, args.apply)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'{len(result)}/6 artifacts verified; apply={args.apply}; report={args.report}')


if __name__ == '__main__':
    main()
