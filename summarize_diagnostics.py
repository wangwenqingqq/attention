"""Stratification, cluster uncertainty and integrity checks for phase-2 exports."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def paired_interval(first, second, kind):
    if set(first) != set(second) or len(first) != 256:
        raise ValueError('Paired sample IDs must match all 256 sequences')
    keys = sorted(first)
    a, b = np.array([first[k] for k in keys]), np.array([second[k] for k in keys])
    rng = np.random.default_rng(11123)
    samples = rng.integers(0, 256, (2000, 256))
    if kind == 'accuracy_pp':
        value, bootstrap = float((a-b).mean()*100), (a[samples]-b[samples]).mean(1)*100
    else:
        value, bootstrap = float(a.mean()/b.mean()), a[samples].mean(1)/b[samples].mean(1)
    return value, np.quantile(bootstrap, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('results', type=Path)
    root = parser.parse_args().results
    route = [json.loads(line) for line in (root/'route_rows.jsonl').read_text().splitlines()]
    oracle = [json.loads(line) for line in (root/'oracle_rows.jsonl').read_text().splitlines()]
    if len(route) != 4*8192 or len(oracle) != 3*2*2*1024:
        raise ValueError('Incomplete declared query denominators')
    signatures = {}
    bins, hits = defaultdict(list), defaultdict(list)
    for r in route:
        identity = (r['sample_id'], r['query_position'])
        source = (r['source_key_position'], r['source_value_position'], r['source_block'])
        if identity in signatures and signatures[identity] != source:
            raise ValueError('Cross-model source metadata drift')
        signatures[identity] = source
        for name, value in [('last_any_hit', str(r['last_any_hit'])), ('last_both_hit', str(r['last_both_hit'])),
                            ('query_position_bin', f"{r['query_position']//64*64}-{r['query_position']//64*64+63}"),
                            ('source_distance_bin', f"{r['source_distance']//64*64}-{r['source_distance']//64*64+63}")]:
            bins[r['group'], name, value].append(r)
        for layer in range(2):
            for head in range(2):
                hits[r['group'], layer, head].append(bool(r['source_hits'][layer][head]))
    write_csv(root/'route_strata.csv', [dict(group=g, stratum=name, value=value, query_count=len(rows),
                                            query_accuracy=float(np.mean([r['correct'] for r in rows])),
                                            query_ce=float(np.mean([r['query_ce'] for r in rows])))
                                       for (g, name, value), rows in sorted(bins.items())])
    write_csv(root/'route_hit_summary.csv', [dict(group=g, layer=layer, head=head, query_count=len(values),
                                                 source_hit_rate=float(np.mean(values)))
                                            for (g, layer, head), values in sorted(hits.items())])
    energy_rows = []
    head_rows = []
    for g in 'BCD':
        for kind in ('query', 'nonquery'):
            for layer in range(2):
                rows = [r for r in oracle if (r['group'], r['kind'], r['layer']) == (g, kind, layer)]
                ids = {(r['sample_id'], r['query_position']) for r in rows}
                if len(ids) != 1024:
                    raise ValueError('Duplicate oracle query IDs')
                for h in range(2):
                    actual, best, zero = [], [], []
                    for r in rows:
                        errors = r['head_candidate_relative_mse'][h]
                        actual_id = r['actual_sets'][h][0]+1 if r['actual_sets'][h] else 0
                        actual.append(errors[actual_id]); best.append(min(errors)); zero.append(errors[0])
                    head_rows.append(dict(group=g, kind=kind, layer=layer, head=h, query_count=1024,
                                          actual_relative_mse=float(np.mean(actual)), oracle_relative_mse=float(np.mean(best)),
                                          zero_relative_mse=float(np.mean(zero))))
                for field in ('joint_target_energy', 'actual_projected_absolute_mse', 'oracle_projected_absolute_mse',
                              'actual_projected_relative_mse', 'oracle_projected_relative_mse'):
                    values = np.array([r[field] for r in rows])
                    energy_rows.append(dict(group=g, kind=kind, layer=layer, field=field, query_count=1024,
                                            minimum=float(values.min()), p01=float(np.quantile(values, .01)),
                                            median=float(np.median(values)), p99=float(np.quantile(values, .99)), maximum=float(values.max())))
    write_csv(root/'oracle_head_summary.csv', head_rows); write_csv(root/'oracle_distributions.csv', energy_rows)
    task = csv_rows(root/'task_sample_rows.csv'); source = csv_rows(root/'source_intervention_sample_rows.csv')
    injected = csv_rows(root/'oracle_task_rows.csv')
    task_map = {(g, ck): {int(r['sample_id']): r for r in task if r['group'] == g and r['checkpoint_kind'] == ck}
                for g in 'FMBCD' for ck in ('best', 'last')}
    comparisons = []
    for ck in ('best', 'last'):
        for metric, kind in [('query_accuracy', 'accuracy_pp'), ('query_ce', 'ce_ratio')]:
            a = {i: float(r[metric]) for i, r in task_map['D', ck].items()}
            b = {i: float(r[metric]) for i, r in task_map['C', ck].items()}
            value, interval = paired_interval(a, b, kind)
            comparisons.append(dict(comparison=f'D-C_{ck}', metric=kind, value=value, lower95=interval[0], upper95=interval[1], sample_count=256))
    inject_summary = []
    for g in 'BCD':
        maps = {}
        for mode in ('baseline', 'joint_oracle_injected'):
            rows = [r for r in injected if r['group'] == g and r['intervention'] == mode]
            if len(rows) != 1024:
                raise ValueError('Incomplete injected task subset')
            by_sample = defaultdict(list)
            for r in rows: by_sample[int(r['sample_id'])].append(r)
            maps[mode] = {i: {'query_accuracy': float(np.mean([r['correct'] == 'True' for r in rr])),
                              'query_ce': float(np.mean([float(r['query_ce']) for r in rr]))} for i, rr in by_sample.items()}
            inject_summary.append(dict(group=g, intervention=mode, query_count=1024,
                                       query_accuracy=float(np.mean([v['query_accuracy'] for v in maps[mode].values()])),
                                       query_ce=float(np.mean([v['query_ce'] for v in maps[mode].values()]))))
        for metric, kind in [('query_accuracy', 'accuracy_pp'), ('query_ce', 'ce_ratio')]:
            value, interval = paired_interval({i: r[metric] for i, r in maps['joint_oracle_injected'].items()},
                                              {i: r[metric] for i, r in maps['baseline'].items()}, kind)
            comparisons.append(dict(comparison=f'{g}_injected-baseline', metric=kind, value=value, lower95=interval[0], upper95=interval[1], sample_count=256))
    for g in 'MBCD':
        for mode in ('source_head0', 'source_head1', 'source_both'):
            for metric, kind in [('query_accuracy', 'accuracy_pp'), ('query_ce', 'ce_ratio')]:
                a = {int(r['sample_id']): float(r[metric]) for r in source if r['group'] == g and r['intervention'] == mode}
                b = {int(r['sample_id']): float(r[metric]) for r in source if r['group'] == g and r['intervention'] == 'baseline'}
                value, interval = paired_interval(a, b, kind)
                comparisons.append(dict(comparison=f'{g}_{mode}-baseline', metric=kind, value=value, lower95=interval[0], upper95=interval[1], sample_count=256))
    write_csv(root/'oracle_task_summary.csv', inject_summary)
    write_csv(root/'paired_cluster_intervals.csv', comparisons)
    manifest = {p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                for p in sorted(root.iterdir()) if p.is_file() and p.name != 'artifact_manifest.json'}
    (root/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(stage='summaries_completed', route_rows=len(route), oracle_rows=len(oracle), bootstrap_samples=256, bootstrap_repetitions=2000)))


if __name__ == '__main__':
    main()
