"""Shared paths and deterministic definitions for the 2026-10-04 experiment."""
from pathlib import Path
import hashlib
import json
import os

EXP = Path(__file__).resolve().parent
REPO = EXP.parent.parent
CONFIG = json.loads((EXP / 'config.json').read_text())
RUNTIME = json.loads((EXP / 'runtime.json').read_text())
TMP = Path(RUNTIME['tmp'])
OLD = REPO / 'research/jpx_arima_learning_20260928/originals/work/arima_grid'
SOURCE = Path(RUNTIME.get('source_dir','/Users/coolguy/Documents/Codex/2026-09-23/new-chat/work/arima_grid'))
SOURCE_MODELS = Path(RUNTIME.get('source_models_dir',str(SOURCE/'models')))
INPUT = TMP / 'inputs'
OUT = TMP / 'new/research/jpx_arima_adf_annual_20261004'
ORDERS = [(p, q) for p in CONFIG['p_candidates'] for q in CONFIG['q_candidates']]
YEARS = CONFIG['years']


def sha(path):
    with Path(path).open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temporary, path)


def hashes():
    return {name: sha(EXP / name) for name in ['experiment_plan.md', 'config.json']}
