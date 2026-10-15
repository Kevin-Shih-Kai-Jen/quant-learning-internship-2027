from pathlib import Path
from unittest.mock import patch
import ast
import contextlib
import importlib
import inspect
import io
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source_snapshot'))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch
from torch import nn
import deep_learning as learning

torch.set_num_threads(1)
results = {}


def capture_error(action):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            action()
    except Exception as error:
        return {'type': type(error).__name__, 'message': str(error)}
    return None


for path in (ROOT / 'source_snapshot').glob('*.py'):
    ast.parse(path.read_text())
results['all_source_syntax_parses'] = True
results['deep_learning_import'] = 'passed'
results['evaluation_import_error'] = capture_error(lambda: importlib.import_module('evaluation'))
results['application_import_error'] = capture_error(lambda: importlib.import_module('JPX_deep_learning_models'))

torch.manual_seed(34)
loader = learning.make_loader([[1], [2], [3], [4]], [0, 0, 0, 0], batch_size=2)
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
    loss = learning.train(model, loader, nn.L1Loss(), optimizer, epochs=20)
    assert step.call_count == 40
    results['twenty_epochs'] = {'optimizer_steps': step.call_count, 'loss_is_finite': bool(np.isfinite(loss))}
weights_before = [parameter.detach().clone() for parameter in model.parameters()]
results['zero_epochs_error'] = capture_error(lambda: learning.train(model, loader, nn.L1Loss(), optimizer, epochs=0))
assert all(torch.equal(before, after) for before, after in zip(weights_before, model.parameters()))
results['zero_epochs_preserves_parameters'] = True

model = nn.Linear(1, 1)
with torch.no_grad():
    model.weight.fill_(1)
    model.bias.zero_()
loss, predictions = learning.evaluation(model, loader, nn.L1Loss())
np.testing.assert_allclose(predictions, [1, 2, 3, 4])
assert loss == 2.5
assert not model.training
results['evaluation_control'] = {'loss': loss, 'predictions': predictions.tolist(), 'eval_mode': True}

payload = io.BytesIO()
original_save, original_load = torch.save, torch.load


def save_memory(value, path):
    original_save(value, payload)


def load_memory(path, **kwargs):
    payload.seek(0)
    return original_load(payload, **kwargs)


with patch.object(learning.Path, 'mkdir'), patch.object(learning.Path, 'is_file', return_value=True), \
     patch.object(torch, 'save', side_effect=save_memory), patch.object(torch, 'load', side_effect=load_memory), \
     contextlib.redirect_stdout(io.StringIO()):
    saved_path = learning.save_model(model, 'synthetic.pt', [2.0], [1.5], '2017-01-01', '2017-07-01', ['Close'])
    loaded, mean, std, columns, start, end = learning.load_model(nn.Linear(1, 1), saved_path)
for name, value in model.state_dict().items():
    torch.testing.assert_close(value, loaded.state_dict()[name])
np.testing.assert_array_equal(mean, [2.0])
np.testing.assert_array_equal(std, [1.5])
assert columns == ['Close'] and start == '2017-01-01' and end == '2017-07-01'
results['save_load_roundtrip'] = {'passed': True, 'serialization': 'real torch serializer, BytesIO only', 'model_path_is_required': inspect.signature(learning.load_model).parameters['model_path'].default is inspect.Parameter.empty}

values = np.array([10.0, 20.0, 30.0, 40.0])
frame = pd.DataFrame({
    'SecuritiesCode': [1001] * 4,
    'Date': pd.date_range('2017-01-01', periods=4).strftime('%Y-%m-%d'),
    'Open': values, 'High': values + 2, 'Low': values - 2, 'Close': values + 1,
    'Volume': values * 100, 'AdjustmentFactor': np.ones(4),
    'ExpectedDividend': np.zeros(4), 'Target': np.arange(4, dtype=float) / 100,
})
columns = ['SecuritiesCode', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor', 'Target']
with patch.object(learning, 'main_loader', side_effect=lambda *args, **kwargs: frame.copy(deep=True)):
    results['duplicate_common_process_error'] = capture_error(lambda: learning.common_process(['2017-01-01', '2017-01-03', '2017-01-03', '2017-01-05'], columns))

tree = ast.parse((ROOT / 'source_snapshot/deep_learning.py').read_text())
common = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'common_process')
return_index = next(i for i, node in enumerate(common.body) if isinstance(node, ast.Return))
results['unreachable_after_first_return'] = {'return_line': common.body[return_index].lineno, 'unreachable_statement_lines': [node.lineno for node in common.body[return_index + 1:]]}
results['tensor_return_annotation_is_factory_not_class'] = learning.make_it_tensor.__annotations__['return'] is torch.tensor

evaluation_tree = ast.parse((ROOT / 'source_snapshot/evaluation.py').read_text())
overall = next(node for node in evaluation_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'overall_performance')
return_keys = [key.value for node in ast.walk(overall) if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict) for key in node.value.keys if isinstance(key, ast.Constant)]
application_tree = ast.parse((ROOT / 'source_snapshot/JPX_deep_learning_models.py').read_text())
lookups = [node.slice.value for node in ast.walk(application_tree) if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'eval_results' and isinstance(node.slice, ast.Constant)]
results['evaluation_result_key_mismatch'] = {'returned': return_keys, 'requested': lookups, 'missing': sorted(set(lookups) - set(return_keys))}
results['scope'] = 'Review only. No production or existing test edits. Synthetic CPU controls and an in-memory checkpoint; no real CSV or full training.'
(ROOT / 'probe_results.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
