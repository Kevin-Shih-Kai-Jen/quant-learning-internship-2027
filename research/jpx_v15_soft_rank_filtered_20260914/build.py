from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    for name in ['model','replay','model_checkpoints','replay_segments']:
        subprocess.run(['clang++','-O3','-std=c++17','-dynamiclib','-framework','Accelerate',str(ROOT/f'{name}.cpp'),'-o',str(ROOT/f'{name}.dylib')],check=True)
