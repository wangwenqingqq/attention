"""Read-only E0/E1/E2 on fresh checkpoint models; final-test stays unopened."""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from block_memory_reference import dense_reference
from mqar_metadata import source_rows, sampled_positions
from oracle_diagnostics import exact_oracle
from train_mqar import build_model, dataset, digest_tensors, mixers


ATTRS = ('k_history', 'capture', 'last_qkv', 'last_selected', 'last_output',
         'forced_selection', 'output_intervention', 'diagnostic_mode',
         'aux_positions', 'auxiliary_loss')


@contextmanager
def preserved(model=None):
    """Restore caller RNG, flags and diagnostic config; assert immutable weights."""
    cpu, numpy, python = torch.get_rng_state(), np.random.get_state(), random.getstate()
    cuda = torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None
    modules = mixers(model) if model is not None else []
    config = [{name: getattr(m, name) for name in ATTRS} for m in modules]
    modes = [(m, m.training) for m in model.modules()] if model is not None else []
    before = digest_tensors(*model.state_dict().values()) if model is not None else None
    try:
        if model is not None:
            model.eval()
            for m in modules:
                m.capture, m.diagnostic_mode = True, True
                m.aux_positions = m.forced_selection = m.output_intervention = None
        yield
    finally:
        if model is not None:
            after = digest_tensors(*model.state_dict().values())
            for m, values in zip(modules, config):
                for name, value in values.items():
                    setattr(m, name, value)
            for m, training in modes:
                m.training = training
        torch.set_rng_state(cpu)
        np.random.set_state(numpy)
        random.setstate(python)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda)
        if model is not None and before != after:
            raise AssertionError("Analysis changed model weights")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    with Path(path).open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def emit(stream, row):
    stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')


def task_rows(logits, y):
    live = y != -100
    return (F.cross_entropy(logits[live], y[live], reduction='none'),
            (logits.argmax(-1)[live] == y[live]).float())


@torch.no_grad()
def e0(model, x, y):
    ce, correct, counts = [], [], dict(detail_pairs=0, summary_pairs=0, router_key_vectors=0)
    modules = mixers(model)
    for start in range(0, len(x), 32):
        xb, yb = x[start:start+32].cuda(), y[start:start+32].cuda()
        loss, ok = task_rows(model(xb), yb)
        ce.extend(loss.tolist()); correct.extend(ok.tolist())
        for b in range(len(xb)):
            pos = torch.where(yb[b] != -100)[0]
            for m in modules:
                q, k, v = m.last_qkv
                if m.group == 'F':
                    counts['detail_pairs'] += int((pos + 1).sum()) * 2
                else:
                    _, stats = m.memory(q[b:b+1, :, pos], k[b:b+1], v[b:b+1], pos, 1, m.group != 'M')
                    for name in counts:
                        counts[name] += int(stats[name])
    return dict(query_ce=float(np.mean(ce)), query_accuracy=float(np.mean(correct)),
                query_count=len(ce), **counts)


@torch.no_grad()
def routes(model, x, y, metadata, stream, group, checkpoint_hash):
    modules = mixers(model)
    intervention = {name: [0., 0., 0] for name in ('baseline', 'replay', 'source_head0', 'source_head1', 'source_both')}
    for start in range(0, len(x), 32):
        xb, yb = x[start:start+32].cuda(), y[start:start+32].cuda()
        logits = model(xb)
        loss, ok = task_rows(logits, yb)
        local = {}
        selected = [m.last_selected for m in modules]
        for b in range(len(xb)):
            pos = torch.where(yb[b] != -100)[0]
            for layer, m in enumerate(modules):
                q, k, v = m.last_qkv
                target = dense_reference(q[b:b+1, :, pos], k[b:b+1], v[b:b+1], pos)
                prediction = m.last_output[b:b+1, :, pos]
                err = (prediction - target).square().mean(-1)[0]
                energy = target.square().mean(-1)[0]
                local[b, layer] = (err / energy.clamp_min(1e-8)).tolist()
        for j, row in enumerate(metadata[start*32:(start+len(xb))*32]):
            b, index = j // 32, j % 32
            position, source = row['query_position'], row['source_block']
            blocks = [[torch.where(s[b, h, position])[0].tolist() for h in range(2)] for s in selected]
            hits = [[source in ids for ids in layer] for layer in blocks]
            emit(stream, dict(group=group, seed=123, checkpoint_sha256=checkpoint_hash, **row,
                              selected_blocks=blocks, source_hits=hits,
                              last_any_hit=any(hits[-1]), last_both_hit=all(hits[-1]),
                              query_ce=float(loss[j]), correct=bool(ok[j]),
                              layer_head_relative_mse=[local[b, layer][0][index:index+1] +
                                                       local[b, layer][1][index:index+1] for layer in range(2)]))
        baseline = logits.detach().clone()
        for name in intervention:
            m = modules[-1]
            if name != 'baseline':
                forced = {}
                for j, row in enumerate(metadata[start*32:(start+len(xb))*32]):
                    b, p, source = j // 32, row['query_position'], row['source_block']
                    heads = (0, 1) if name in ('replay', 'source_both') else ((0,) if name == 'source_head0' else (1,))
                    for h in heads:
                        forced[b, h, p] = (torch.where(selected[-1][b, h, p])[0].tolist()
                                             if name == 'replay' else (source,))
                m.forced_selection = forced
                current = model(xb)
                m.forced_selection = None
                if name == 'replay':
                    torch.testing.assert_close(current, baseline, atol=3e-6, rtol=3e-5)
                else:
                    torch.testing.assert_close(current[yb == -100], baseline[yb == -100], atol=3e-6, rtol=3e-5)
            else:
                current = baseline
            ce, acc = task_rows(current, yb)
            intervention[name][0] += float(ce.sum())
            intervention[name][1] += float(acc.sum())
            intervention[name][2] += len(ce)
    return [dict(group=group, seed=123, checkpoint_sha256=checkpoint_hash, intervention=name,
                 query_ce=sums[0]/sums[2], query_accuracy=sums[1]/sums[2], query_count=sums[2])
            for name, sums in intervention.items()]


@torch.no_grad()
def oracles(model, x, y, samples, stream, group, checkpoint_hash):
    rows, injected = [], []
    modules = mixers(model)
    for sample, (true, other) in enumerate(samples):
        xb = x[sample:sample+1].cuda()
        baseline = model(xb)
        captured = [m.last_qkv for m in modules]
        patches = {}
        for layer, (m, (q, k, v)) in enumerate(zip(modules, captured)):
            for kind, cpu_pos in (('query', true), ('nonquery', other)):
                pos = cpu_pos.cuda()
                result = exact_oracle(m.memory, q[:, :, pos], k, v, pos, m.out_proj.weight)
                n = result['output'].shape[0]
                for j, p in enumerate(cpu_pos.tolist()):
                    actual_ids = result['actual_ids'][:, j].tolist()
                    head_ids = result['head_best'][:, j].tolist()
                    joint_ids = result['joint_ids'][:, j].tolist()
                    valid_n = int(result['valid'][:, j].sum())
                    row = dict(group=group, seed=123, checkpoint_sha256=checkpoint_hash,
                               sample_id=sample, query_position=p, kind=kind, layer=layer,
                               actual_sets=[([] if i == 0 else [i-1]) for i in actual_ids],
                               head_best_sets=[([] if i == 0 else [i-1]) for i in head_ids],
                               joint_best_sets=[([] if i == 0 else [i-1]) for i in joint_ids],
                               candidate_sets=[([] if i == 0 else [i-1]) for i in range(valid_n)],
                               head_candidate_absolute_mse=result['head_absolute'][:valid_n, :, j].T.tolist(),
                               head_candidate_relative_mse=result['head_relative'][:valid_n, :, j].T.tolist(),
                               head_target_energy=result['head_energy'][:, j].tolist(),
                               joint_target_energy=float(result['joint_energy'][j]),
                               actual_projected_absolute_mse=float(result['actual_absolute'][j]),
                               actual_projected_relative_mse=float(result['actual_relative'][j]),
                               oracle_projected_absolute_mse=float(result['joint_absolute'][j]),
                               oracle_projected_relative_mse=float(result['joint_relative'][j]),
                               zero_projected_relative_mse=float(result['zero_relative'][j]),
                               zero_projected_absolute_mse=float(result['zero_absolute'][j]),
                               candidate_detail_pairs=result['counts']['detail_pairs'][:valid_n, j].tolist(),
                               candidate_summary_pairs=result['counts']['summary_pairs'][:valid_n, j].tolist(),
                               router_key_vectors=int(result['counts']['router_key_vectors'][j]),
                               k_history=1, joint_used_blocks=[int(i != 0) for i in joint_ids])
                    if row['oracle_projected_absolute_mse'] > row['zero_projected_absolute_mse'] + 3e-6:
                        raise AssertionError('At-most-one oracle lost the empty candidate')
                    emit(stream, row); rows.append(row)
                if kind == 'query' and layer == len(modules)-1:
                    for j, p in enumerate(cpu_pos.tolist()):
                        for h in range(2):
                            patches[0, h, p] = result['best_output'][h, j]
        modules[-1].output_intervention = patches
        oracle_logits = model(xb)
        modules[-1].output_intervention = None
        unpatched = torch.ones(x.shape[1], dtype=torch.bool, device=xb.device)
        unpatched[true.cuda()] = False
        torch.testing.assert_close(oracle_logits[:, unpatched], baseline[:, unpatched], atol=3e-6, rtol=3e-5)
        for name, logits in (('baseline', baseline), ('joint_oracle_injected', oracle_logits)):
            target = y[sample, true].cuda()
            loss = F.cross_entropy(logits[0, true.cuda()], target, reduction='none')
            correct = logits[0, true.cuda()].argmax(-1) == target
            for j, p in enumerate(true.tolist()):
                injected.append(dict(group=group, sample_id=sample, query_position=p,
                                     intervention=name, query_ce=float(loss[j]), correct=bool(correct[j])))
        if sample % 32 == 0:
            print(json.dumps(dict(stage='oracle', group=group, sample=sample)), flush=True)
    return rows, injected


def bootstrap_ratio(d, c):
    rng = np.random.default_rng(11123)
    ratios = []
    for _ in range(2000):
        ids = rng.integers(0, len(c), len(c))
        ratios.append(float(d[ids].mean()/c[ids].mean()))
    return [float(np.quantile(ratios, .025)), float(np.quantile(ratios, .975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True, help='Directory containing all five seed-123 runs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    with preserved():
        x, y = dataset(512, 32, 256, 9123)
        samples = sampled_positions(y, torch.Generator().manual_seed(10123))
        metadata = source_rows(x, y)
        data_hash = digest_tensors(x, y)
        validations = dataset(512, 32, 1000, 2123)
        valid_hash = digest_tensors(*validations)
        del validations
        provenance = dict(dev_seed=9123, dev_examples=256, dev_sha256=data_hash,
                          query_sampling_seed=10123, original_valid_sha256=valid_hash,
                          analysis_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                          source_sha256={p.name: sha(p) for p in Path('.').glob('*.py')},
                          torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                          tf32=False, final_test_generated=False, checkpoints=[])
        same, source_interventions, all_oracles, injection = [], [], [], []
        with (args.output/'route_rows.jsonl').open('x') as route_stream, (args.output/'oracle_rows.jsonl').open('x') as oracle_stream:
            for group in 'FMBCD':
                run = args.runs/f'{group}_seed123'
                meta = json.loads((run/'metadata.json').read_text())
                for key, expected in dict(group=group, seed=123, steps=40000, train_examples=100000,
                                          dropout=.1, dtype='float32', batch=32, aux_weight=.1,
                                          aux_interval=8, aux_queries=16, aux_layers=1, lr=.001).items():
                    if meta[key] != expected:
                        raise ValueError(f'Frozen metadata mismatch: {key}')
                if meta['valid_sha256'] != valid_hash:
                    raise ValueError('Original validation generator/hash drift')
                histories = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
                for kind in ('best', 'last'):
                    checkpoint = run/f'{kind}.pt'
                    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    if payload['step'] != (min(histories, key=lambda r: r['query_ce'])['step'] if kind == 'best' else 40000):
                        raise ValueError('Checkpoint selection rule mismatch')
                    checkpoint_hash = sha(checkpoint)
                    model = build_model(group, 123, 'cuda', meta['dropout'])
                    model.load_state_dict(payload['model'], strict=True)
                    del payload
                    with preserved(model):
                        result = e0(model, x, y)
                        same.append(dict(group=group, seed=123, checkpoint_kind=kind,
                                         checkpoint_sha256=checkpoint_hash, step=(min(histories, key=lambda r: r['query_ce'])['step'] if kind == 'best' else 40000), **result))
                        print(json.dumps(same[-1]), flush=True)
                        if kind == 'best' and group != 'F':
                            source_interventions.extend(routes(model, x, y, metadata, route_stream, group, checkpoint_hash))
                            if group != 'M':
                                rows, injected = oracles(model, x, y, samples, oracle_stream, group, checkpoint_hash)
                                all_oracles.extend(rows); injection.extend(injected)
                    provenance['checkpoints'].append(dict(group=group, kind=kind, sha256=checkpoint_hash,
                                                          state_before_after_equal=True, metadata=meta,
                                                          raw_sha256={name: sha(run/name) for name in ('metadata.json', 'metrics.jsonl', 'train.jsonl', 'diagnostics.jsonl', 'result.json', 'initial.pt')}))
                    del model
        if digest_tensors(x, y) != data_hash:
            raise AssertionError('Input/label mutation')
    write_csv(args.output/'same_budget_dev.csv', same)
    write_csv(args.output/'source_intervention.csv', source_interventions)
    write_csv(args.output/'oracle_task_rows.csv', injection)
    summary = []
    for group in 'BCD':
        for kind in ('query', 'nonquery'):
            for layer in range(2):
                rows = [r for r in all_oracles if r['group'] == group and r['kind'] == kind and r['layer'] == layer]
                summary.append(dict(group=group, kind=kind, layer=layer, query_count=len(rows),
                                    **{field: float(np.mean([r[field] for r in rows])) for field in
                                       ('actual_projected_absolute_mse', 'actual_projected_relative_mse',
                                        'oracle_projected_absolute_mse', 'oracle_projected_relative_mse',
                                        'zero_projected_relative_mse', 'joint_target_energy')},
                                    zero_selected_head_fraction=float(np.mean([i == 0 for r in rows for i in r['joint_used_blocks']]))))
    write_csv(args.output/'oracle_summary.csv', summary)
    arrays = {g: np.array([np.mean([r['oracle_projected_relative_mse'] for r in all_oracles
                                   if r['group'] == g and r['kind'] == 'query' and r['layer'] == 1 and r['sample_id'] == s])
                           for s in range(256)]) for g in 'CD'}
    ratio = float(arrays['D'].mean()/arrays['C'].mean())
    interval = bootstrap_ratio(arrays['D'], arrays['C'])
    task = {g: next(r for r in same if r['group'] == g and r['checkpoint_kind'] == 'best') for g in 'CD'}
    gate = ratio <= .9 and interval[1] < 1 and task['D']['query_accuracy'] >= task['C']['query_accuracy']-.01 and task['D']['query_ce'] <= 1.01*task['C']['query_ce']
    provenance.update(oracle_D_C_ratio=ratio, paired_sample_bootstrap_95=interval, E3_gate=bool(gate),
                      optimizer_created=False, synthetic_data=True)
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(stage='completed', E3_gate=bool(gate), oracle_ratio=ratio, ratio_95=interval)), flush=True)


if __name__ == '__main__':
    main()
