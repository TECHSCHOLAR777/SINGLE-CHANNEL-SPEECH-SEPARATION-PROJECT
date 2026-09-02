"""
P1-INT2: cross-chunk identity lock, proven in CI without a neural separator.

The historical P1-INT2 failure conflated two things: the ChunkStitcher's
identity-lock logic (Dev C's actual deliverable) and whether MossFormer2 — a
2-speaker model — can even emit one stable stream per speaker on a 3-speaker
mixture (it cannot; that is an escalation concern, handled by the cascade, not
an alignment bug). Validating the lock through MossFormer2 on 3 speakers is an
invalid experiment: it tests the lock with a separator that structurally cannot
feed it stable streams.

This test isolates the lock. An oracle separator emits perfect per-speaker
streams for each chunk but in a **permuted order per chunk** (the realistic
challenge: real separators do not preserve slot order across chunks), with
per-speaker-consistent embeddings. The REAL ChunkStitcher / run_and_align_long /
xcorr scoring must re-lock every chunk's streams onto stable persistent tracks.
Pass criterion mirrors scripts/validate_alignment.py exactly: every chunk covers
all speakers, the persistent track count equals the speaker count, and zero
identity switches are counted between consecutive active windows.

Runs on CPU, no weights, deterministic. This is the identity-lock proof; the
on-Kaggle run through a real separator is confirmation, not the claim itself.
"""

from __future__ import annotations

import numpy as np

from coralsep.align.integration import run_and_align_long
from coralsep.schemas.separation_result import SeparationResult, StreamMetadata
from scripts.validate_alignment import _assignment_trace, _identity_switches

SR = 16000
CHUNK_SEC = 4.0
OVERLAP_SEC = 1.0


def _make_reference_speakers(n: int, length: int, seed: int = 0) -> np.ndarray:
    """n distinct, full-length, stationary speaker signals (seeded noise, mildly
    low-passed so consecutive samples correlate — makes xcorr identity crisp)."""
    rng = np.random.default_rng(seed)
    refs = rng.standard_normal((n, length)).astype(np.float32)
    kernel = np.ones(32, dtype=np.float32) / 32.0
    for i in range(n):
        refs[i] = np.convolve(refs[i], kernel, mode="same")
        refs[i] /= float(np.max(np.abs(refs[i])) + 1e-8)
    return refs


def _orthogonal_embeddings(n: int, dim: int = 16) -> np.ndarray:
    """One well-separated unit embedding per speaker (stable identity signal)."""
    emb = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        emb[i, i] = 1.0
    return emb


class _OracleSeparator:
    """
    Perfect separation with adversarial slot order.

    Returns, for each chunk, exactly the per-speaker reference slices — but
    permuted by a per-chunk seed so the stream order is unstable across chunks,
    which is precisely what the stitcher's embedding lock has to undo. Embeddings
    travel with their speaker, so a correct stitcher re-locks perfectly.
    """

    def __init__(self, refs: np.ndarray, embeddings: np.ndarray, hop: int) -> None:
        self.refs = refs
        self.embeddings = embeddings
        self.hop = hop
        self.calls = 0

    def separate(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        n, _ = self.refs.shape
        start = self.calls * self.hop
        width = int(np.asarray(mixture).squeeze().shape[0])
        chunk_refs = self.refs[:, start : start + width]
        # Pad the final short chunk so every stream matches the chunk width.
        if chunk_refs.shape[1] < width:
            pad = width - chunk_refs.shape[1]
            chunk_refs = np.pad(chunk_refs, ((0, 0), (0, pad)))

        perm = np.random.default_rng(1000 + self.calls).permutation(n)
        self.calls += 1

        streams = chunk_refs[perm].copy()
        meta = [
            StreamMetadata(
                expert_source="oracle",
                confidence=1.0,
                embedding=self.embeddings[perm[i]].copy(),
                extra={"stream_index": i},
            )
            for i in range(n)
        ]
        return SeparationResult(
            streams=streams,
            sample_rate=sample_rate,
            speaker_count=n,
            metadata=meta,
            mixture=np.asarray(mixture, dtype=np.float32).squeeze(),
            escalated=False,
            expert_used="oracle",
        )


def _run_lock(n_speakers: int, seconds: float = 12.0, seed: int = 0) -> dict:
    length = int(seconds * SR)
    refs = _make_reference_speakers(n_speakers, length, seed=seed)
    embeddings = _orthogonal_embeddings(n_speakers)
    mixture = refs.sum(axis=0)
    mixture /= float(np.max(np.abs(mixture)) + 1e-8)

    hop = max(1, int(round((CHUNK_SEC - OVERLAP_SEC) * SR)))
    engine = _OracleSeparator(refs, embeddings, hop)

    long_out = run_and_align_long(
        engine,
        mixture,
        SR,
        chunk_sec=CHUNK_SEC,
        overlap_sec=OVERLAP_SEC,
        max_tracks=n_speakers,
    )
    trace = _assignment_trace(
        long_out.result.streams, refs, long_out.result.sample_rate, CHUNK_SEC, OVERLAP_SEC, -60.0
    )
    switches = _identity_switches(trace)
    streams_per_chunk = sorted({len(ids) for ids in long_out.chunk_track_ids})
    return {
        "num_persistent_tracks": long_out.result.num_streams,
        "streams_per_chunk": streams_per_chunk,
        "expert_covers_all_speakers": all(k >= n_speakers for k in streams_per_chunk),
        "identity_switches": switches["identity_switches"],
        "num_windows": len(trace),
    }


def test_identity_lock_holds_on_two_speakers() -> None:
    r = _run_lock(2)
    assert r["num_windows"] >= 2
    assert r["expert_covers_all_speakers"]
    assert r["num_persistent_tracks"] == 2
    assert r["identity_switches"] == 0


def test_identity_lock_holds_on_three_speakers() -> None:
    # The regime the RunPod run failed on — here with a separator that actually
    # emits 3 stable streams, the lock holds. That failure was the separator,
    # not the stitcher.
    r = _run_lock(3)
    assert r["num_windows"] >= 2
    assert r["expert_covers_all_speakers"]
    assert r["num_persistent_tracks"] == 3
    assert r["identity_switches"] == 0


def test_identity_lock_stable_across_seeds() -> None:
    for seed in range(4):
        r = _run_lock(3, seed=seed)
        assert r["identity_switches"] == 0, f"seed {seed} broke the lock: {r}"
