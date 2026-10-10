"""Read-only report of CC2 backend and coherent-capture activation."""
from __future__ import annotations

import json
from pathlib import Path


def report(folder):
    folder = Path(folder)
    config = folder / 'config.json'
    overlay = folder / 'assistant_overlay.py'
    settings = (json.loads(config.read_text(encoding='utf-8'))
                if config.exists() else {})
    source = overlay.read_text(encoding='utf-8') if overlay.exists() else ''
    mode = settings.get('solver_engine', 'coldclear2')
    print('Configured engine:', mode)
    print('Cold Clear adapter available:',
          (folder / 'solver_cc2.py').is_file())
    print('Coherent capture:', 'CoherentFrame(' in source)
    print('Pose dropout guard:', 'pose_guard.keep(' in source)
    print('Fail-closed CC2 mode:', 'COLD CLEAR UNAVAILABLE' in source)
    if mode != 'coldclear2':
        print('ERROR: solver_engine must be coldclear2.')
    if 'find_best_cc2_fast' not in source:
        print('WARNING: overlay does not select the validated CC2 adapter.')


if __name__ == '__main__':
    report(Path.cwd())
