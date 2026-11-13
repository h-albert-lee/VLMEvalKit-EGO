"""Frozen input utilities for the label-adaptation comparison."""
from pathlib import Path
import hashlib
import json
import os

from egoown_cvpr.core import LABELS, parse_answer, metrics


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def readl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as handle:
        handle.write(canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def writel(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as handle:
        for row in rows:
            handle.write(canonical(row) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as handle:
        handle.write(canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stimulus(record):
    fields = ('model_key', 'model', 'revision', 'provider', 'backend', 'parameters',
              'condition', 'variant', 'options')
    value = {key: record[key] for key in fields}
    value['messages'] = [dict(role=m['role'], content=[
        {k: v for k, v in part.items() if k != 'path'} for part in m['content']])
        for m in record['messages']]
    return value


def seal(record):
    result = dict(record)
    result['input_fingerprint'] = sha(canonical(stimulus(record)))
    result['request_key'] = sha(canonical({key: result[key] for key in
        ('suite_id', 'id', 'input_fingerprint')}))
    return result


def checked_image(root, part):
    root = Path(root).resolve()
    path = (root / part['path']).resolve()
    require(path.is_relative_to(root), 'Image outside suite')
    data = path.read_bytes()
    require(sha(data) == part['sha256'], 'Frozen image changed')
    return path, data


def verify_freeze(root):
    root = Path(root)
    freeze = load(root / 'freeze.json')
    for relative, expected in freeze['hashes'].items():
        require(sha((root / relative).read_bytes()) == expected, 'Changed frozen file: ' + relative)
    return freeze
