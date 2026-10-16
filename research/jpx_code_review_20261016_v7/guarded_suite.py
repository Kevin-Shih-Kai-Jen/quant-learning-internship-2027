"""Run unchanged regression tests with a trace guard for nonadvancing splits.

The guard raises after observing the same split three times. It never changes
source files or repairs the loop, so dependent tests report the blocking bug.
"""
from pathlib import Path
import json
import os
import runpy
import sys

ROOT=Path(__file__).resolve().parent
CODES=ROOT/'source_snapshot'
sys.path.insert(0,str(CODES))
os.environ['JPX_CODES_DIR']=str(CODES)
import utils
records=[]

def trace_calls(frame,event,arg):
    if event!='call' or frame.f_code is not utils.create_start_end_date.__code__:
        return None
    steps=[]
    def trace_lines(current,event,arg):
        if event=='line' and current.f_lineno==49:
            values=current.f_locals
            key=(values['train_end'],values['val_start'],values['val_end'])
            steps.append(tuple(x.isoformat() for x in key))
            if len(steps)>=3 and steps[-1]==steps[-2]==steps[-3]:
                records.append({'observed_splits':steps,'ans_length':len(values['ans'])})
                raise RuntimeError('Review guard: create_start_end_date repeats the same split without advancing train_end')
        return trace_lines
    return trace_lines

sys.argv=[str(ROOT/'tests_snapshot/run_tests.py'),'--codes',str(CODES),'--report-dir',str(ROOT/'regression_results')]
sys.settrace(trace_calls)
try:
    runpy.run_path(str(ROOT/'tests_snapshot/run_tests.py'),run_name='__main__')
finally:
    sys.settrace(None)
    (ROOT/'loop_guard_events.json').write_text(json.dumps(records,indent=2)+'\n')
