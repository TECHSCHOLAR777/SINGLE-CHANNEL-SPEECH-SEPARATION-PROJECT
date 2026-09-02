"""Tests for the ChunkStitcher track cap (P1-C3 phantom-track fix)."""

from __future__ import annotations

import numpy as np
import pytest

from coralsep.align.chunking import ChunkStitcher

SR = 16000
D = 32
CHUNK = 4.0
OVERLAP = 1.0
T = int(CHUNK * SR)


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def _stitcher(max_tracks: int | None) -> ChunkStitcher:
    return ChunkStitcher(
        sample_rate=SR,
        chunk_sec=CHUNK,
        overlap_sec=OVERLAP,
        match_threshold=0.35,
        ema=0.7,
        max_tracks=max_tracks,
    )


def _streams(k: int) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((k, T)).astype(np.float32) * 0.01


def test_uncapped_stitcher_mints_a_phantom_track_per_chunk():
    """The original bug: an unstable second slot spawns a new track every chunk."""
    rng = np.random.default_rng(1)
    stable = _unit(rng.standard_normal(D))
    drifting = [_unit(rng.standard_normal(D)) for _ in range(3)]

    stitcher = _stitcher(max_tracks=None)
    ids = [stitcher.add_chunk(_streams(2), np.stack([stable, drifting[i]])) for i in range(3)]

    assert ids == [[0, 1], [0, 2], [0, 3]]
    assert stitcher.num_tracks == 4  # 4 tracks for what should be at most 3 speakers


def test_cap_prevents_phantom_tracks():
    """Same drifting input, capped: the bank never exceeds the expected count."""
    rng = np.random.default_rng(1)
    stable = _unit(rng.standard_normal(D))
    drifting = [_unit(rng.standard_normal(D)) for _ in range(3)]

    stitcher = _stitcher(max_tracks=3)
    for i in range(3):
        stitcher.add_chunk(_streams(2), np.stack([stable, drifting[i]]))

    assert stitcher.num_tracks == 3
    assert stitcher.forced_matches >= 1  # the third chunk had to be forced


def test_forced_match_does_not_poison_the_track_embedding():
    """A match that failed the threshold must not be folded into the track EMA."""
    rng = np.random.default_rng(2)
    a = _unit(rng.standard_normal(D))
    b = _unit(rng.standard_normal(D))
    intruder = _unit(rng.standard_normal(D))

    stitcher = _stitcher(max_tracks=2)
    stitcher.add_chunk(_streams(2), np.stack([a, b]))
    before = [track.embedding.copy() for track in stitcher._tracks]

    # Bank is full; this embedding matches nothing, so it must be force-assigned.
    stitcher.add_chunk(_streams(2), np.stack([a, intruder]))

    assert stitcher.num_tracks == 2
    assert stitcher.forced_matches == 1
    # track 1 was force-matched, so its embedding is unchanged
    assert np.allclose(stitcher._tracks[1].embedding, before[1])


def test_cap_still_matches_normally_when_embeddings_are_clean():
    """The cap must not fire when the threshold is already doing its job."""
    rng = np.random.default_rng(3)
    a = _unit(rng.standard_normal(D))
    b = _unit(rng.standard_normal(D))

    stitcher = _stitcher(max_tracks=2)
    first = stitcher.add_chunk(_streams(2), np.stack([a, b]))
    second = stitcher.add_chunk(_streams(2), np.stack([b, a]))  # permuted, same speakers

    assert first == [0, 1]
    assert second == [1, 0]  # identity lock survives the permutation
    assert stitcher.num_tracks == 2
    assert stitcher.forced_matches == 0


def test_streams_in_one_chunk_never_collide_on_one_track():
    """Hungarian is one-to-one, and forcing must not break that."""
    rng = np.random.default_rng(4)
    bank = [_unit(rng.standard_normal(D)) for _ in range(3)]

    stitcher = _stitcher(max_tracks=3)
    stitcher.add_chunk(_streams(3), np.stack(bank))
    # three mutually unmatched embeddings, bank at cap: all three must be forced
    # onto three DISTINCT tracks, not collapsed onto one
    intruders = np.stack([_unit(rng.standard_normal(D)) for _ in range(3)])
    ids = stitcher.add_chunk(_streams(3), intruders)

    assert len(set(ids)) == 3
    assert stitcher.num_tracks == 3


def test_cap_of_one_collapses_everything_to_a_single_track():
    rng = np.random.default_rng(5)
    stitcher = _stitcher(max_tracks=1)
    for _ in range(3):
        stitcher.add_chunk(_streams(1), _unit(rng.standard_normal(D))[None])
    assert stitcher.num_tracks == 1


def test_uncapped_behaviour_is_unchanged_by_default():
    """max_tracks=None must reproduce the pre-fix behaviour exactly."""
    rng = np.random.default_rng(6)
    a = _unit(rng.standard_normal(D))
    b = _unit(rng.standard_normal(D))
    stitcher = ChunkStitcher(sample_rate=SR, chunk_sec=CHUNK, overlap_sec=OVERLAP)
    assert stitcher.max_tracks is None
    stitcher.add_chunk(_streams(2), np.stack([a, b]))
    assert stitcher.num_tracks == 2
    assert stitcher.forced_matches == 0


def test_invalid_cap_rejected():
    with pytest.raises(ValueError):
        _stitcher(max_tracks=0)


def test_stitch_shape_survives_the_cap():
    rng = np.random.default_rng(7)
    a = _unit(rng.standard_normal(D))
    stitcher = _stitcher(max_tracks=2)
    for _ in range(3):
        stitcher.add_chunk(_streams(2), np.stack([a, _unit(rng.standard_normal(D))]))
    out = stitcher.stitch()
    assert out.shape[0] == 2
    assert np.all(np.isfinite(out))
