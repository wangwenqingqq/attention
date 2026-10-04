"""Curate complete MQAR runs and paired initialization-seed evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paired_interval(values):
    if len(values) != 3:
        raise ValueError("This registered interval requires exactly three paired seeds")
    mean = statistics.mean(values)
    radius = 4.302652729696142 * statistics.stdev(values) / math.sqrt(3)
    return {"mean": mean, "ci95": [mean - radius, mean + radius], "n": 3,
            "estimator": "Student-t seed-effects interval, df=2; exploratory normality assumption"}


def read_rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def validate_contract(groups):
    assert set(groups) == set("FMBCD"), "Missing control group"
    for key in ("initial_state_sha256", "train_sha256", "valid_sha256", "batch_order_sha256"):
        assert len({row[key] for row in groups.values()}) == 1, f"Unmatched {key}"
    assert groups["C"]["aux_schedule_sha256"] == groups["D"]["aux_schedule_sha256"], "Unmatched auxiliary sampling"
    assert groups["F"]["best_validation"]["query_accuracy"] >= 0.9, "Failed full-baseline admission"


def validate_history(history, best):
    assert [row["step"] for row in history] == [1, *range(250, 40001, 250)], "Incomplete evaluation history"
    for row in history:
        assert row["query_count"] == 32000, "Unexpected validation denominator"
        assert row["training_tokens"] == row["step"] * 32 * 512
        assert math.isfinite(row["query_ce"]) and row["query_ce"] >= 0
        assert 0 <= row["query_accuracy"] <= 1
    assert best == min(history, key=lambda row: row["query_ce"]), "Best checkpoint/history mismatch"


def collect_campaign(campaign):
    full = list(campaign.glob("F_seed*/metadata.json"))
    if len(full) != 1:
        raise ValueError("Each campaign must contain exactly one complete initialization seed")
    seed = json.loads(full[0].read_text())["seed"]
    groups = {}
    for group in "FMBCD":
        run = campaign / f"{group}_seed{seed}"
        metadata = json.loads((run / "metadata.json").read_text())
        result = json.loads((run / "result.json").read_text())
        assert metadata["seed"] == seed and metadata["group"] == result["group"] == group
        assert result["status"] == "completed" and result["steps"] == metadata["steps"] == 40000
        assert result["initial_state_sha256"] == metadata["initial_state_sha256"]
        for key, expected in {"train_examples": 100000, "dropout": 0.1, "lr": 0.001,
                              "batch": 32, "aux_weight": 0.1, "aux_interval": 8,
                              "aux_queries": 16, "aux_layers": 1, "parameters": 806400,
                              "dtype": "float32", "torch": "2.11.0+cu130",
                              "torchvision": "0.26.0+cu130", "cuda": "13.0",
                              "upstream_commit": "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb",
                              "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}.items():
            assert metadata[key] == expected, f"Unexpected {group} {key}"
        history = read_rows(run / "metrics.jsonl")
        validate_history(history, result["best_validation"])
        order, auxiliary = hashlib.sha256(), hashlib.sha256()
        samples, aux_samples, early, late = 0, 0, [], []
        for line in (run / "train.jsonl").open():
            row = json.loads(line)
            samples += 1
            assert row["step"] == samples
            assert all(math.isfinite(row[key]) for key in ("task_loss", "aux_loss", "grad_norm_preclip"))
            order.update((row["batch_ids_sha256"] + "\n").encode())
            if row["aux_layer"] is not None:
                aux_samples += 1
                assert group in "CD" and row["step"] % 8 == 0 and row["aux_layer"] in (0, 1)
                assert len(row["aux_positions"]) == len(set(row["aux_positions"])) == 16
                auxiliary.update(json.dumps([row["aux_layer"], row["aux_positions"]],
                                            separators=(",", ":")).encode() + b"\n")
                if row["step"] <= 1000:
                    early.append(row["aux_loss"])
                if row["step"] > 39000:
                    late.append(row["aux_loss"])
        assert samples == 40000
        assert aux_samples == (5000 if group in "CD" else 0)
        groups[group] = {key: metadata[key] for key in (
            "initial_state_sha256", "train_sha256", "valid_sha256", "source_commit",
            "upstream_commit", "torch", "torchvision", "cuda", "gpu", "parameters")}
        groups[group].update(
            best_validation=result["best_validation"], final_validation=history[-1],
            training_steps=samples, auxiliary_steps=aux_samples, training_tokens=40000 * 32 * 512,
            batch_order_sha256=order.hexdigest(), aux_schedule_sha256=auxiliary.hexdigest(),
            auxiliary_mean_first_1000_steps=statistics.mean(early) if early else None,
            auxiliary_mean_last_1000_steps=statistics.mean(late) if late else None,
            wall_seconds_including_validation_and_diagnostics=result["wall_seconds"],
            timing_claim="observed reference cost, not paired kernel timing or system speedup",
            validation_history=history, ood_budget_context_rows=read_rows(run / "diagnostics.jsonl"),
            artifacts={name: {"sha256": sha256(run / name), "bytes": (run / name).stat().st_size}
                       for name in ("initial.pt", "best.pt", "last.pt", "train.jsonl", "metrics.jsonl", "diagnostics.jsonl")})
    validate_contract(groups)
    source = campaign.parents[1]
    sources = {name: sha256(source / name) for name in ("train_mqar.py", "memory_attention.py")}
    references = [source / "block_memory_reference.py", source / "original" / "block_memory_reference.py",
                  source / "attention_memory_starter" / "block_memory_reference.py"]
    hashes = {sha256(path) for path in references if path.is_file()}
    assert len(hashes) == 1, "Missing or conflicting reference source"
    sources["block_memory_reference.py"] = hashes.pop()
    return {"seed": seed, "groups": groups, "scientific_source_sha256": sources}


def validate_replicates(records):
    assert sorted(row["seed"] for row in records) == [123, 124, 125]
    for key in ("train_sha256", "valid_sha256", "batch_order_sha256"):
        assert len({row["groups"]["F"][key] for row in records}) == 1, f"Cross-seed mismatch: {key}"
    assert len({row["groups"]["C"]["aux_schedule_sha256"] for row in records}) == 1, "Cross-seed auxiliary sampling mismatch"
    sources = [row["scientific_source_sha256"] for row in records]
    assert all(source == sources[0] for source in sources), "Cross-seed scientific source mismatch"
    assert len({row["groups"]["F"]["initial_state_sha256"] for row in records}) == 3, "Seeds did not change initialization"


def summarize(campaigns):
    records = [collect_campaign(path) for path in campaigns]
    records.sort(key=lambda row: row["seed"])
    validate_replicates(records)
    effects = {}
    for candidate, keeper in (("D", "C"), ("C", "B")):
        ratios, accuracy = [], []
        for record in records:
            a, b = (record["groups"][g]["best_validation"] for g in (candidate, keeper))
            ratios.append(math.log(a["query_ce"] / b["query_ce"]))
            accuracy.append(100 * (a["query_accuracy"] - b["query_accuracy"]))
        log_ratio = paired_interval(ratios)
        effects[f"{candidate}_vs_{keeper}"] = {
            "per_seed_log_ce_ratio": ratios, "per_seed_accuracy_delta_pp": accuracy,
            "geometric_ce_ratio": math.exp(log_ratio["mean"]),
            "ce_ratio_ci95": [math.exp(value) for value in log_ratio["ci95"]],
            "accuracy_delta_pp": paired_interval(accuracy), "paired_log_ce_ratio": log_ratio}
    return {"scope": "three initialization/dropout seeds on fixed synthetic data; validation-selected checkpoints",
            "denominator": "CE/accuracy over 32000 held-out validation query labels per evaluation",
            "system_speed_claim": False, "kv_storage_saving_claim": False,
            "campaigns": records, "paired_effects": effects,
            "remaining_gates": ["independent external training reproduction", "budget-trained Pareto points",
                                "real-corpus/aggregation tasks", "optimized kernels and complete external comparators"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.campaign)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
