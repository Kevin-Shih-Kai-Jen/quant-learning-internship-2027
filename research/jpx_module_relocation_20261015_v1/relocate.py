"""Prepare a versioned, logic-preserving relocation; never writes the live source."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path('/Users/coolguy/Desktop/JPX/codes')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def definition(source, name):
    node = next(node for node in ast.parse(source).body if getattr(node, 'name', None) == name)
    return ast.get_source_segment(source, node)


def replace_exact(source, old, new, count):
    assert source.count(old) == count, (old, source.count(old), count)
    return source.replace(old, new)


def main():
    before_dir, after_dir = HERE / 'before', HERE / 'after'
    before_dir.mkdir()
    after_dir.mkdir()
    before = {p.name: p.read_bytes() for p in sorted(SOURCE.glob('*.py'))}
    assert not before['models.py'].strip(), 'models.py has changed; stop before overwriting work'
    assert 'utils.py' not in before, 'utils.py already exists; review before proceeding'
    after = dict(before)
    dl = before['deep_learning.py'].decode()
    model_class = definition(dl, 'CNN_LSTM')
    common = definition(dl, 'common_process')
    timer = definition(dl, 'print_running_time')
    retained = ['make_it_tensor', 'make_loader', 'train', 'evaluation', 'save_model', 'load_model']
    imports = ('import torch\nfrom torch import nn\n'
               'from torch.utils.data import TensorDataset, DataLoader\n'
               'import pandas as pd\nimport numpy as np\nfrom pathlib import Path')
    after['deep_learning.py'] = (imports + '\n\n\n' + '\n\n\n'.join(definition(dl, name) for name in retained) + '\n').encode()
    after['models.py'] = ('import torch\nfrom torch import nn\n\n\n' + model_class + '\n').encode()
    # Binding time to its module matches the existing, unchanged timer body.
    after['utils.py'] = ('import time\n\n\n' + timer + '\n').encode()

    experiment = before['JPX_deep_learning_models.py'].decode()
    experiment = replace_exact(experiment, 'import evaluation\n', 'import evaluation\nfrom data_loader import main_loader\nimport data_processor\nimport utils\n', 1)
    experiment = replace_exact(experiment, '@deep_learning.print_running_time', common + '\n\n\n@utils.print_running_time', 1)
    experiment = replace_exact(experiment, 'deep_learning.common_process(', 'common_process(', 2)
    experiment = replace_exact(experiment, 'deep_learning.models(', 'models.CNN_LSTM(', 1)
    after['JPX_deep_learning_models.py'] = experiment.encode()
    entry = replace_exact(before['main.py'].decode(),
                          'from JPX_deep_learning_models import model_2, load_model, FEAUTRES',
                          'from JPX_deep_learning_models import model_2, FEAUTRES\nfrom deep_learning import load_model', 1)
    after['main.py'] = entry.encode()
    for name, data in before.items():
        (before_dir / name).write_bytes(data)
    for name, data in after.items():
        ast.parse(data, filename=name)
        (after_dir / name).write_bytes(data)
    changed = [name for name in after if before.get(name) != after[name]]
    manifest = {
        'date': '2026-10-15',
        'source_directory': str(SOURCE),
        'scope': 'Move definitions and repair module imports/call targets only. Function bodies and training/data/evaluation/checkpoint algorithms are preserved.',
        'moved_definitions': {
            'deep_learning.CNN_LSTM': 'models.CNN_LSTM',
            'deep_learning.common_process': 'JPX_deep_learning_models.common_process',
            'deep_learning.print_running_time': 'utils.print_running_time',
        },
        'additional_wiring_changes': [
            'deep_learning.models(...) -> models.CNN_LSTM(...)',
            'main imports load_model from deep_learning',
            'utils imports the time module used by the unchanged timer body',
        ],
        'changed_files': changed,
        'before': {name: {'bytes': len(data), 'sha256': digest(data)} for name, data in before.items()},
        'after': {name: {'bytes': len(data), 'sha256': digest(data)} for name, data in after.items()},
        'existing_behavior_preserved': [
            'train epoch parameter remains unused',
            'normalization still fits before date splitting',
            'model and optimizer are still shared across folds',
            'model_2 save_model call still lacks feature_columns and passes mean=0/std=1',
            'load_model still has model_path=str and use_loaded_model still passes an empty path',
            'evaluation metrics, dates, model architecture, feature order, and checkpoint format remain unchanged',
            'early stopping is not added',
        ],
    }
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'changed_files': changed, 'snapshot_directory': str(HERE)}, indent=2))


if __name__ == '__main__':
    main()
