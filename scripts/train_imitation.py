"""E8 reproducible teacher collection, hand-disjoint validation, GPU training."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from trucoai.agents.learned import collect_chunk, export_policy
from trucoai.rules import ONLINE_O1


def validation_mask(groups, mode='modulo'):
    if mode == 'modulo':
        return groups % 10 == 0
    if mode != 'hash':
        raise ValueError('unknown split mode')
    selected = [g for g in np.unique(groups)
                if int.from_bytes(hashlib.sha256(str(int(g)).encode()).digest()[:8], 'big') % 10 == 0]
    return np.isin(groups, selected)


def load_initial(net, path):
    import torch
    from trucoai.agents.learned import PolicyNetwork
    source = PolicyNetwork(path)
    layers = [layer for layer in net if isinstance(layer, torch.nn.Linear)]
    if len(layers) != len(source.layers):
        raise ValueError('initial network architecture mismatch')
    with torch.no_grad():
        for layer, (w, b) in zip(layers, source.layers):
            if tuple(layer.weight.shape) != tuple(w.T.shape):
                raise ValueError('initial network shape mismatch')
            layer.weight.copy_(torch.from_numpy(w.T.copy()))
            layer.bias.copy_(torch.from_numpy(b.copy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--hands', type=int, default=30000)
    ap.add_argument('--start-seed', type=int, default=300000)
    ap.add_argument('--seed', type=int, default=20261020)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--teacher', default='zap:v1')
    ap.add_argument('--data', help='Reuse an immutable synthetic dataset')
    ap.add_argument('--split-mode', choices=('modulo', 'hash'), default='modulo')
    ap.add_argument('--init', help='Optional immutable policy weights for warm start')
    args = ap.parse_args()
    out = ROOT / 'runs' / args.run_id
    out.mkdir(exist_ok=False)
    start = time.time()
    sources = list((ROOT / 'trucoai').rglob('*.py')) + [Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    config = {'args': vars(args), 'rules_fingerprint': ONLINE_O1.fingerprint,
              'code_sha256': hashes, 'hypothesis': 'Learning the specified teacher actions improves rollout fidelity and posterior search.',
              'provenance': 'Synthetic independent hands; teacher only receives Observation; no human data or privileged labels.',
              'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    if args.data:
        with np.load(args.data) as z:
            data = {k: z[k].copy() for k in ('x', 'y', 'mask', 'group')}
    else:
        chunks = [(args.teacher, args.start_seed + i, min(100, args.hands - i))
                  for i in range(0, args.hands, 100)]
        parts = []
        with ProcessPoolExecutor(args.workers) as ex:
            for j, part in enumerate(ex.map(collect_chunk, chunks)):
                parts.append(part)
                if (j + 1) % 10 == 0:
                    print(f'collected {min((j+1)*100,args.hands)}/{args.hands} hands; {time.time()-start:.1f}s', flush=True)
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        with (out / 'data.npz').open('xb') as f:
            np.savez_compressed(f, **data)
        del parts
    # Entire independent hands are split. No labels from game test seeds enter training.
    valid = validation_mask(data['group'], args.split_mode)
    config['dataset_sha256'] = hashlib.sha256(Path(args.data or out/'data.npz').read_bytes()).hexdigest()
    config['split_audit'] = {'train_groups': int(len(np.unique(data['group'][~valid]))),
                            'validation_groups': int(len(np.unique(data['group'][valid]))),
                            'overlap': int(len(np.intersect1d(data['group'][~valid], data['group'][valid])))}
    config['dataset'] = {'rows': len(valid), 'hands': len(np.unique(data['group'])),
                         'validation_rows': int(valid.sum()), 'features': data['x'].shape[1],
                         'label_counts': np.bincount(data['y'], minlength=9).tolist(),
                         'missing_nonfinite': int((~np.isfinite(data['x'])).sum()),
                         'group_min': int(data['group'].min()), 'group_max': int(data['group'].max())}
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = torch.nn.Sequential(torch.nn.Linear(data['x'].shape[1], 96), torch.nn.ReLU(),
                              torch.nn.Linear(96, 96), torch.nn.ReLU(), torch.nn.Linear(96, 9)).to(device)
    if args.init:
        load_initial(net, args.init)
        config['initial_policy_sha256'] = hashlib.sha256(Path(args.init).read_bytes()).hexdigest()
    x = torch.tensor(data['x'], device=device)
    y = torch.tensor(data['y'], device=device)
    mask = torch.tensor(data['mask'], device=device)
    tr = torch.tensor(np.flatnonzero(~valid), device=device)
    va = torch.tensor(np.flatnonzero(valid), device=device)
    optim = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.001)
    best, best_epoch, history = float('inf'), -1, []
    state = None
    for epoch in range(args.epochs):
        net.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        losses = []
        for ids in perm.split(1024):
            logits = net(x[ids]).masked_fill(~mask[ids], -1e9)
            loss = torch.nn.functional.cross_entropy(logits, y[ids])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            losses.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():
            logits = net(x[va]).masked_fill(~mask[va], -1e9)
            vl = float(torch.nn.functional.cross_entropy(logits, y[va]))
            acc = float((logits.argmax(1) == y[va]).float().mean())
            card = y[va] < 6
            cardacc = float((logits.argmax(1)[card] == y[va][card]).float().mean())
        history.append({'epoch': epoch, 'train_loss': sum(losses)/len(losses), 'val_loss': vl,
                        'val_accuracy': acc, 'val_card_accuracy': cardacc})
        if vl < best:
            best, best_epoch = vl, epoch
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if epoch % 5 == 0:
            print(history[-1], flush=True)
        if epoch - best_epoch >= 12:
            break
    net.load_state_dict(state)
    net.cpu()
    meta = {'rules_fingerprint': ONLINE_O1.fingerprint, 'run_id': args.run_id,
            'teacher': args.teacher, 'seed': args.seed, 'best_epoch': best_epoch, 'val_loss': best}
    export_policy(net, out / 'policy.npz', meta)
    from trucoai.agents.learned import PolicyNetwork
    portable = PolicyNetwork(out / 'policy.npz')
    xx = data['x'][:100]
    actual = xx
    for i, (w, b) in enumerate(portable.layers):
        actual = actual @ w + b
        if i + 1 < len(portable.layers):
            actual = np.maximum(actual, 0)
    with torch.no_grad():
        error = float(np.max(np.abs(actual - net(torch.tensor(xx)).numpy())))
    assert error < 1e-4, error
    config.update({'history': history, 'best_epoch': best_epoch, 'export_max_abs_error': error,
                   'elapsed_s': time.time()-start, 'device': device, 'torch': torch.__version__,
                   'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
                   'policy_sha256': hashlib.sha256((out/'policy.npz').read_bytes()).hexdigest()})
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'SAVED {out} best={best:.5f} epoch={best_epoch} elapsed={time.time()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
