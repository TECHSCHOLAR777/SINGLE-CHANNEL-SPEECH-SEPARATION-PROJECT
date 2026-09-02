"""
Full evaluation matrix (Dev C, P5-C1).

Runs CoRAL-Sep against the 8-condition × 4-N fixed eval set produced by
src/coralsep/data/fixed_eval_generator.py and returns a structured results table.

Conditions (8):
  clean | reverb | noise | codec | reverb+noise | reverb+codec
  | noise+codec | reverb+noise+codec

N ∈ {2, 3, 4, 5} speakers.

Per-mixture metrics:
  - SI-SDRi (relative to unprocessed mixture)
  - SDRi
  - DNSMOS OVRL (reference-free, 16 kHz output)
  - Speaker count accuracy (N_hat == N_true)
  - Completeness probability (from head)
  - OOD flag rate

Outputs:
  - A flat JSON-L file with one record per mixture.
  - A summary CSV aggregated by (condition, N).
  - Bootstrap CIs via eval/stats.py.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_CONDITIONS = (
    "clean",
    "reverb",
    "noise",
    "codec",
    "reverb+noise",
    "reverb+codec",
    "noise+codec",
    "reverb+noise+codec",
)

_N_VALUES = (2, 3, 4, 5)


@dataclass
class MixtureRecord:
    """One row in the evaluation result table."""

    mixture_id: str
    condition: str
    n_true: int
    n_hat: int
    si_sdri: float
    sdri: float
    dnsmos_ovrl: float | None
    completeness_prob: float
    ood_flag: bool
    gate_vector: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def count_correct(self) -> bool:
        return self.n_hat == self.n_true


@dataclass
class EvalMatrix:
    """Aggregated evaluation results."""

    records: list[MixtureRecord] = field(default_factory=list)

    def append(self, record: MixtureRecord) -> None:
        self.records.append(record)

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for rec in self.records:
                f.write(json.dumps(asdict(rec)) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "EvalMatrix":
        obj = cls()
        with open(str(path)) as f:
            for line in f:
                d = json.loads(line)
                obj.records.append(MixtureRecord(**d))
        return obj

    def summary_by_condition(self) -> dict[str, dict]:
        """
        Returns nested dict:
          summary[condition][n] = {si_sdri_mean, sdri_mean, dnsmos_mean,
                                   count_acc, n_mixtures}
        """
        from collections import defaultdict
        buckets: dict[tuple, list] = defaultdict(list)
        for rec in self.records:
            buckets[(rec.condition, rec.n_true)].append(rec)

        summary = {}
        for (cond, n), recs in sorted(buckets.items()):
            if cond not in summary:
                summary[cond] = {}
            si_sdri_vals = [r.si_sdri for r in recs]
            sdri_vals = [r.sdri for r in recs]
            dnsmos_vals = [r.dnsmos_ovrl for r in recs if r.dnsmos_ovrl is not None]
            count_acc = float(np.mean([r.count_correct for r in recs]))
            summary[cond][n] = {
                "si_sdri_mean": float(np.mean(si_sdri_vals)),
                "sdri_mean": float(np.mean(sdri_vals)),
                "dnsmos_mean": float(np.mean(dnsmos_vals)) if dnsmos_vals else None,
                "count_acc": count_acc,
                "n_mixtures": len(recs),
            }
        return summary

    def to_summary_csv(self, path: str | Path) -> None:
        import csv
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.summary_by_condition()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "condition", "n", "si_sdri_mean", "sdri_mean",
                "dnsmos_mean", "count_acc", "n_mixtures"
            ])
            writer.writeheader()
            for cond, n_dict in summary.items():
                for n, stats in n_dict.items():
                    writer.writerow({"condition": cond, "n": n, **stats})


def run_eval_matrix(
    pipeline,
    eval_manifest: str | Path,
    dnsmos_scorer=None,
    max_per_bucket: int | None = None,
    device: str = "cpu",
) -> EvalMatrix:
    """
    Run the CoRAL-Sep pipeline over all mixtures in eval_manifest.

    Args:
        pipeline: CoralSepPipeline instance.
        eval_manifest: Path to JSONL manifest produced by fixed_eval_generator.py.
            Each line has {mixture_id, condition, n_true, mixture_path,
            reference_paths}.
        dnsmos_scorer: DnsmosScorer for 16 kHz quality scoring.
        max_per_bucket: Limit mixtures per (condition, N) for quick ablations.
        device: Torch device.

    Returns:
        EvalMatrix with one record per mixture.
    """
    import soundfile as sf
    from coralsep.train.losses import si_snr as _si_snr_torch
    import torch

    matrix = EvalMatrix()
    bucket_counts: dict[tuple, int] = {}

    with open(str(eval_manifest)) as f:
        for line in f:
            item = json.loads(line)
            mixture_id = item["mixture_id"]
            condition = item["condition"]
            n_true = int(item["n_true"])

            key = (condition, n_true)
            if max_per_bucket is not None and bucket_counts.get(key, 0) >= max_per_bucket:
                continue

            try:
                mix_wav, sr = sf.read(item["mixture_path"], dtype="float32", always_2d=False)
                refs = [sf.read(p, dtype="float32", always_2d=False)[0] for p in item["reference_paths"]]
            except Exception as e:
                print(f"[eval] skip {mixture_id}: {e}")
                continue

            try:
                result = pipeline.run(mix_wav, sr)
            except Exception as e:
                print(f"[eval] pipeline error {mixture_id}: {e}")
                continue

            # SI-SDRi / SDRi (oracle permutation via PIT).
            si_sdri_val, sdri_val = _compute_sisdr_sdri(result.streams_8k, refs, sr, device)

            # DNSMOS (16 kHz output).
            dnsmos_ovrl = None
            if dnsmos_scorer is not None and dnsmos_scorer.is_available:
                try:
                    scores = dnsmos_scorer.score(
                        result.streams_16k.mean(0),  # mix of separated streams
                        sample_rate=16000,
                    )
                    dnsmos_ovrl = scores["ovrl"]
                except Exception:
                    pass

            record = MixtureRecord(
                mixture_id=mixture_id,
                condition=condition,
                n_true=n_true,
                n_hat=result.speaker_count,
                si_sdri=si_sdri_val,
                sdri=sdri_val,
                dnsmos_ovrl=dnsmos_ovrl,
                completeness_prob=result.completeness_prob,
                ood_flag=result.ood_flag,
                gate_vector=result.gate_vector,
            )
            matrix.append(record)
            bucket_counts[key] = bucket_counts.get(key, 0) + 1

    return matrix


def _compute_sisdr_sdri(
    separated: np.ndarray,
    references: list[np.ndarray],
    sample_rate: int,
    device: str = "cpu",
) -> tuple[float, float]:
    """Oracle PIT SI-SDRi and SDRi (dB improvement over mixture)."""
    import torch
    K = separated.shape[0]
    R = len(references)
    n = separated.shape[1]

    # Align lengths.
    t = min(n, min(len(r) for r in references))
    sep = torch.from_numpy(separated[:, :t]).float()
    ref_list = [torch.from_numpy(r[:t]).float() for r in references]

    # Match K and R by zero-padding shorter set.
    while len(ref_list) < K:
        ref_list.append(torch.zeros(t))
    refs_t = torch.stack(ref_list[:K])  # (K, T)

    best = -999.0
    best_sdr = -999.0
    from itertools import permutations
    for perm in permutations(range(K)):
        val = float(_si_snr_batch(sep[list(perm)], refs_t).mean().item())
        sdr_val = float(_sdr_batch(sep[list(perm)], refs_t).mean().item())
        if val > best:
            best = val
            best_sdr = sdr_val

    # Baseline: unprocessed mixture (just the first ref for si-snr baseline).
    mix_baseline = refs_t.mean(0, keepdim=True).expand(K, -1)
    baseline_sisdr = float(_si_snr_batch(mix_baseline, refs_t).mean().item())
    baseline_sdr = float(_sdr_batch(mix_baseline, refs_t).mean().item())

    return best - baseline_sisdr, best_sdr - baseline_sdr


def _si_snr_batch(est: "torch.Tensor", ref: "torch.Tensor") -> "torch.Tensor":
    import torch
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    dot = (est * ref).sum(dim=-1, keepdim=True)
    ref_pow = (ref * ref).sum(dim=-1, keepdim=True) + 1e-8
    s_target = dot / ref_pow * ref
    noise = est - s_target
    return 10 * torch.log10((s_target**2).sum(-1) / ((noise**2).sum(-1) + 1e-8) + 1e-8)


def _sdr_batch(est: "torch.Tensor", ref: "torch.Tensor") -> "torch.Tensor":
    import torch
    dot = (est * ref).sum(-1)
    ref_pow = (ref * ref).sum(-1) + 1e-8
    noise = est - dot.unsqueeze(-1) / ref_pow.unsqueeze(-1) * ref
    return 10 * torch.log10((ref * ref).sum(-1) / ((noise * noise).sum(-1) + 1e-8) + 1e-8)
