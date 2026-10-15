"""Check unchanged definitions and before/after synthetic behavior without real JPX data."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


class CanonicalNames(ast.NodeTransformer):
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'deep_learning':
            if node.attr == 'common_process':
                return ast.Name(id='common_process', ctx=node.ctx)
            if node.attr == 'models':
                return ast.Attribute(value=ast.Name(id='models', ctx=ast.Load()), attr='CNN_LSTM', ctx=node.ctx)
            if node.attr == 'print_running_time':
                return ast.Attribute(value=ast.Name(id='utils', ctx=ast.Load()), attr='print_running_time', ctx=node.ctx)
        return node


def definitions(root):
    result = {}
    for path in sorted(root.glob('*.py')):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                result[f'{path.stem}.{node.name}'] = node
    return result


RUNTIME = r'''
import contextlib, hashlib, importlib, inspect, io, json, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
layout = sys.argv[2]
import numpy as np
import pandas as pd
import torch
import deep_learning as dl
import JPX_deep_learning_models as experiment
torch.set_num_threads(1)
torch.manual_seed(20261015)
if layout == 'before':
    cls = dl.CNN_LSTM
    common = dl.common_process
    owner = dl
else:
    import models, main, utils
    cls = models.CNN_LSTM
    common = experiment.common_process
    owner = experiment
    assert main.CNN_LSTM is models.CNN_LSTM
    assert main.load_model is dl.load_model
    called = []
    with contextlib.redirect_stdout(io.StringIO()):
        utils.print_running_time(lambda: called.append(True))()
    assert called == [True]
model = cls(num_features=2, cnn_out_channels=4, kernel_size=3, hidden_size=5)
model.eval()
X = np.arange(48, dtype=np.float32).reshape(4, 6, 2) / 48
y = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
loader = dl.make_loader(X, y, batch_size=2)
with torch.no_grad():
    initial = model(torch.from_numpy(X)).numpy().tolist()
class CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.01)
        self.steps = 0
    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)
optimizer = CountingSGD(model.parameters())
train_loss = dl.train(model, loader, torch.nn.L1Loss(), optimizer, epoch=20, device='cpu')
valid_loss, predictions = dl.evaluation(model, loader, torch.nn.L1Loss(), device='cpu')
buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
buffer.seek(0)
restored = cls(num_features=2, cnn_out_channels=4, kernel_size=3, hidden_size=5)
restored.load_state_dict(torch.load(buffer, weights_only=True))
restored_loss, restored_predictions = dl.evaluation(restored, loader, torch.nn.L1Loss(), device='cpu')
assert np.array_equal(predictions, restored_predictions)
assert valid_loss == restored_loss
source = pd.DataFrame({
    'SecuritiesCode':[1001]*4,
    'Date':['2017-01-04','2017-01-05','2017-07-03','2017-07-04'],
    'Open':[10.,20.,30.,40.], 'High':[10.,20.,30.,40.],
    'Low':[10.,20.,30.,40.], 'Close':[10.,20.,30.,40.],
    'Volume':[100.,200.,300.,400.], 'ExpectedDividend':[0.,0.,0.,0.],
    'AdjustmentFactor':[1.,1.,1.,1.], 'Target':[0.01,0.02,-0.01,-0.02],
})
split = ('2017-01-01','2017-07-01','2017-07-01','2018-01-01')
with patch.object(owner, 'main_loader', return_value=source.copy()), patch.object(torch.backends.mps, 'is_available', return_value=False), contextlib.redirect_stdout(io.StringIO()):
    train_df, valid_df, device = common(split, experiment.COL_NAMES)
try:
    inspect.signature(dl.save_model).bind(model=model, model_name='CNN_LSTM', mean=0, std=1, train_start=split[0], train_end=split[1])
except TypeError as exc:
    preserved_save_error = str(exc)
else:
    raise AssertionError('The existing save call mismatch was unexpectedly changed')
print(json.dumps({
    'initial_predictions':initial,
    'train_loss':train_loss,
    'validation_loss':valid_loss,
    'trained_predictions':predictions.tolist(),
    'trained_state_sha256':hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest(),
    'state_dict_round_trip':True,
    'optimizer_steps_for_epoch_20':optimizer.steps,
    'train_preprocessing':train_df.to_json(date_format='iso'),
    'valid_preprocessing':valid_df.to_json(date_format='iso'),
    'device':str(device),
    'existing_save_call_error':preserved_save_error,
}))
'''


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    before, after = definitions(HERE / 'before'), definitions(HERE / 'after')
    mapping = manifest['moved_definitions']
    checked = []
    for old_name, old_node in before.items():
        new_name = mapping.get(old_name, old_name)
        normalized = CanonicalNames().visit(copy.deepcopy(old_node))
        assert ast.dump(normalized) == ast.dump(after[new_name]), (old_name, new_name)
        checked.append({'before': old_name, 'after': new_name})
    assert set(after) == {mapping.get(name, name) for name in before}
    for name, metadata in manifest['before'].items():
        assert hashlib.sha256((HERE / 'before' / name).read_bytes()).hexdigest() == metadata['sha256']
    for name, metadata in manifest['after'].items():
        assert hashlib.sha256((HERE / 'after' / name).read_bytes()).hexdigest() == metadata['sha256']
    results = {}
    for layout in ['before', 'after']:
        run = subprocess.run([sys.executable, '-B', '-c', RUNTIME, str(HERE / layout), layout], check=True, capture_output=True, text=True)
        results[layout] = json.loads(run.stdout)
    assert results['before'] == results['after'], 'Synthetic behavior differs after relocation'
    report = {
        'status': 'passed',
        'python': sys.version,
        'definitions_checked': checked,
        'definition_count': len(checked),
        'allowed_changes': 'Only the three explicitly mapped module references are normalized before AST comparison.',
        'all_after_modules_import': True,
        'timer_module_binding': 'verified',
        'synthetic_before_after_equal': True,
        'synthetic_results': results['after'],
        'limitations': 'No real JPX dataset read or full training run. Existing training/preprocessing/checkpoint-call issues intentionally preserved.',
    }
    (HERE / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status':report['status'], 'definition_count':len(checked), 'synthetic_before_after_equal':True, 'all_after_modules_import':True}, indent=2))


if __name__ == '__main__':
    main()
