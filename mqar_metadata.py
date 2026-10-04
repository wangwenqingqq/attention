"""Source annotations from the original KV prefix; never a deployable input."""
import torch


def source_rows(inputs, labels, pairs=32):
    if inputs.shape != labels.shape or inputs.ndim != 2:
        raise ValueError("Expected matching [sample, position] inputs and labels")
    rows = []
    for sample, (x, y) in enumerate(zip(inputs, labels)):
        keys = x[:2 * pairs:2].tolist()
        if len(set(keys)) != pairs:
            raise ValueError("Original prefix keys must be unique")
        lookup = {key: i for i, key in enumerate(keys)}
        positions = torch.where(y != -100)[0].tolist()
        if len(positions) != pairs:
            raise ValueError("Unexpected query count")
        for position in positions:
            if position < 2 * pairs or int(x[position]) not in lookup:
                raise ValueError("Query has no original prefix source")
            pair = lookup[int(x[position])]
            value_position = 2 * pair + 1
            if int(x[value_position]) != int(y[position]):
                raise ValueError("Source value and target disagree")
            rows.append(dict(sample_id=sample, query_position=position,
                             source_key_position=2 * pair,
                             source_value_position=value_position,
                             source_block=value_position // 32,
                             source_distance=position - value_position, pair_index=pair))
    return rows


def sampled_positions(labels, generator):
    """Four true and four nonqueries per example, with a private generator."""
    result = []
    for y in labels:
        query = torch.where(y != -100)[0]
        other = torch.where(y == -100)[0]
        result.append((query[torch.randperm(len(query), generator=generator)[:4]].sort().values,
                       other[torch.randperm(len(other), generator=generator)[:4]].sort().values))
    return result
