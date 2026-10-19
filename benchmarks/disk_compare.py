"""Small isolated-process resident AdamW versus two-pass disk measurements.

Run from project root: python benchmarks/disk_compare.py --device cuda
This intentionally allocates only a debug model; results are not7B estimates.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from morph import MorphConfig
from morph.training.runner import train_run
from streamtrain.disk import disk_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--mode', choices=['resident', 'disk'])
    args = parser.parse_args()
    if args.mode is None:
        outputs = {}
        for mode in ['resident', 'disk']:
            raw = subprocess.check_output([sys.executable, __file__, '--device', args.device, '--mode', mode], text=True)
            outputs[mode] = json.loads(raw)
        print(json.dumps(outputs, indent=2))
        return
    torch.set_num_threads(1)
    cfg = MorphConfig.from_file('configs/morph_debug.json')
    tokens = torch.tensor(list(Path('docs/smoke.txt').read_bytes()))[None]
    if args.mode == 'resident':
        import resource
        result = train_run(cfg, tokens, 3, device=args.device, optimizer_name='adamw',
                           sequence_length=8, accumulation=2)
        metrics = result['metrics']
        metrics['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        metrics['final_loss'] = result['losses'][-1]
    else:
        with tempfile.TemporaryDirectory(prefix='morph-disk-bench-') as parent:
            metrics = disk_train(cfg, tokens, Path(parent)/'state', 3, device=args.device,
                                 sequence_length=8, accumulation=2)
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
