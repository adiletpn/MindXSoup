"""Experimental full recomputation with reentrant checkpointing; disposable checkout only.

Restores the existing source even on failure. Never changes the default configuration.
The first pass records no autograd graph inside checkpointed layers. Inputs without
gradients retain the original non-reentrant path, which can train parameters in that case.
"""

from pathlib import Path
import re
import subprocess
import sys


def main():
    repo = Path(__file__).resolve().parents[1]
    path = repo / 'solution/recompute.py'
    original = path.read_text(encoding='utf-8')
    candidate = re.sub(r'^BUDGET_TOKENS = .*$', 'BUDGET_TOKENS = 0', original, flags=re.MULTILINE)
    old = 'use_reentrant=False, preserve_rng_state=True'
    if candidate.count(old) != 1:
        raise RuntimeError('Unexpected checkpoint source; refusing to patch')
    candidate = candidate.replace(old,
                                  'use_reentrant=hidden_states.requires_grad, preserve_rng_state=True')
    try:
        path.write_text(candidate, encoding='utf-8')
        for command in (
            ['-m', 'pytest', 'solution', '-q'],
            ['harness/run_task.py', '--repeats', '2'],
            ['harness/check_rules.py'],
        ):
            subprocess.run([sys.executable, *command], cwd=repo, check=True)
    finally:
        path.write_text(original, encoding='utf-8')


if __name__ == '__main__':
    main()
