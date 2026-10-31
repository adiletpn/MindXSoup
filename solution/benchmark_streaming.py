"""Same-machine ABBA comparison. Run as a script in a disposable Kaggle checkout.

Uses the unchanged official harness, cold caches per run, and seed 7 by default.
Temporarily edits streaming constants; always restores the original source.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=Path, default=Path('/kaggle/working/out/stream_abba'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = repo / 'solution/stream.py'
    original = source.read_text(encoding='utf-8')
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        subprocess.run([sys.executable, '-m', 'pytest', 'solution', '-q'], cwd=repo, check=True)
        for number, candidate in enumerate((False, True, True, False)):
            variant = 'candidate' if candidate else 'baseline'
            text = re.sub(r'^SLOTS = \d+$', f'SLOTS = {2 if candidate else 3}',
                          original, flags=re.MULTILINE)
            text = re.sub(r'^COALESCE_TRANSFERS = (True|False)$',
                          f'COALESCE_TRANSFERS = {candidate}', text, flags=re.MULTILINE)
            source.write_text(text, encoding='utf-8')
            out = args.out / f'{number}_{variant}'
            subprocess.run([sys.executable, 'harness/run_task.py', '--only', 'solution',
                            '--seed', str(args.seed), '--out', str(out)], cwd=repo, check=True)
            result = json.loads((out / 'result.json').read_text(encoding='utf-8'))
            results.append({'variant': variant,
                            'stream_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                            'result': result})
            (args.out / 'comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            if (not result['summary']['solution'].get('ok')
                    or result.get('evals', {}).get('solution', {}).get('exit_code') != 0):
                raise RuntimeError(f'{variant} failed; inspect {out}')
    finally:
        source.write_text(original, encoding='utf-8')


if __name__ == '__main__':
    main()
