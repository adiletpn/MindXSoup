"""Cold-process startup diagnostics; writes JSON plus raw importtime output."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(mode):
    import torch
    if mode == 'patches':
        import transformers
        from triton.runtime.jit import JITFunction
    import solution
    torch.cuda.init()
    if 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Run on a Kaggle T4')
    if mode == 'patches':
        timings = {}
        for name, fn in list(vars(solution).items()):
            if name.startswith('patch_') and callable(fn):
                def wrap(*args, _fn=fn, _name=name, **kwargs):
                    start = time.perf_counter()
                    try:
                        return _fn(*args, **kwargs)
                    finally:
                        timings[_name] = time.perf_counter() - start
                setattr(solution, name, wrap)
        start = time.perf_counter()
        solution.apply()
        timings['apply_total'] = time.perf_counter() - start
        solution.warmup.wait()
        timings['apply_plus_warmup_join'] = time.perf_counter() - start
        return timings
    from solution.rmsnorm import rmsnorm_fwd_kernel, rmsnorm_bwd_kernel
    from solution.rope import _rope_launch
    from solution.fused_ce import launch_rows
    from solution.nf4 import unpack_nf4, _nf4_table
    from solution.swiglu import example_swiglu_fwd_kernel, example_swiglu_bwd_kernel
    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(shape, device='cuda', dtype=dtype)
    x, w, y, r, dx = zeros(2048, 1536), zeros(1536), zeros(2048, 1536), zeros(2048), zeros(2048, 1536)
    q = zeros(4, 512, 12, 128).transpose(1, 2)
    k = zeros(4, 512, 2, 128).transpose(1, 2)
    c, s = torch.ones((4, 512, 128), device='cuda'), zeros(4, 512, 128)
    logits = zeros(512, 151936, dtype=torch.float16)
    target, loss, entropy, argmax = zeros(512, dtype=torch.int32), zeros(512), zeros(512), zeros(512, dtype=torch.int32)
    n = 8960 * 1536 // 2
    packed, scales = zeros(n, dtype=torch.uint8), zeros(n // 32, dtype=torch.uint8)
    code, scale2, offset = zeros(256), zeros((n // 32 + 255) // 256), zeros()
    _nf4_table(packed.device)
    gate, up, out, dg, du = [zeros(2048, 8960, dtype=torch.float16) for _ in range(5)]
    count = gate.numel()
    jobs = {
        'rmsnorm_fwd_kernel': lambda: rmsnorm_fwd_kernel[(2048,)](x,w,y,r,1536,1e-6,BLOCK=2048,num_warps=8),
        'rmsnorm_bwd_kernel': lambda: rmsnorm_bwd_kernel[(2048,)](y,x,w,r,dx,1536,BLOCK=2048,num_warps=8),
        'rope_kernel': lambda: _rope_launch(False,q,k,c,s,q.dtype,k.dtype),
        'head_ce_rows_kernel': lambda: launch_rows(logits,target,loss,entropy,argmax,True),
        'nf4_unpack_kernel': lambda: unpack_nf4(packed,(8960,1536),scale2,64,torch.float16,absmax_8bit=scales,code=code,offset=offset),
        'example_swiglu_fwd_kernel': lambda: example_swiglu_fwd_kernel[((count+1023)//1024,)](gate,up,out,count,BLOCK=1024),
        'example_swiglu_bwd_kernel': lambda: example_swiglu_bwd_kernel[((count+1023)//1024,)](out,gate,up,dg,du,count,BLOCK=1024),
    }
    torch.cuda.synchronize()
    if mode in ('serial', 'threads'):
        selected = [jobs[n] for n in ('rmsnorm_fwd_kernel','rope_kernel','head_ce_rows_kernel')]
        start = time.perf_counter()
        if mode == 'threads':
            def run(fn):
                torch.cuda.set_device(0)
                fn()
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(run, selected))
        else:
            for fn in selected:
                fn()
        torch.cuda.synchronize()
        return {'seconds': time.perf_counter()-start}
    result = {}
    for name, fn in jobs.items():
        samples = []
        for _ in range(2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append(time.perf_counter()-start)
        result[name] = samples
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/kaggle/working/out/startup'))
    parser.add_argument('--worker', choices=['patches','kernels','serial','threads'])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        result = worker(args.worker)
        (args.out / (args.worker+'.json')).write_text(json.dumps(result, indent=2))
        return
    data = {}
    with tempfile.TemporaryDirectory(prefix='startup-import-') as cache:
        env = dict(os.environ, TRITON_CACHE_DIR=cache)
        proc = subprocess.run([sys.executable,'-X','importtime','-c','import solution'],cwd=ROOT,env=env,capture_output=True,text=True)
        (args.out/'importtime.txt').write_text(proc.stderr)
        proc.check_returncode()
        rows = []
        for line in proc.stderr.splitlines():
            if line.startswith('import time:'):
                parts = line[len('import time:'):].split('|')
                if len(parts)==3 and parts[0].strip().isdigit():
                    rows.append({'module':parts[2].strip(),'self_s':int(parts[0])/1e6,'cumulative_s':int(parts[1])/1e6})
        data['imports_top15_self'] = sorted(rows,key=lambda r:-r['self_s'])[:15]
        data['imports_top15_cumulative'] = sorted(rows,key=lambda r:-r['cumulative_s'])[:15]
    for mode in ('patches','kernels','serial','threads','threads','serial','serial','threads'):
        with tempfile.TemporaryDirectory(prefix='startup-'+mode+'-') as cache:
            env = dict(os.environ, TRITON_CACHE_DIR=cache)
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',mode,'--out',str(args.out.resolve())],cwd=ROOT,env=env,check=True)
            value = json.loads((args.out/(mode+'.json')).read_text())
            if mode in ('serial', 'threads'):
                data.setdefault(mode, []).append(value)
            else:
                data[mode] = value
    (args.out/'startup_probe.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data,indent=2))


if __name__ == '__main__':
    main()
