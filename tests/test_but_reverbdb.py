"""Regression for I-052: the BUT ReverbDB download URL pointed at the wrong host.

data/prepare/but_reverbdb.py claimed BUT ReverbDB was OpenSLR resource 17 and
built its download URL from https://www.openslr.org/resources/17/. That
resource is MUSAN, an unrelated corpus; both filenames the old code tried
(BUT_ReverbDB_rel_19_06_RIR.tgz and reverb_data_but.zip) 404 there, and always
have. The real archive is hosted on BUT's own server. This had zero test
coverage, which is exactly how a URL that never worked survived undetected.
"""

from __future__ import annotations

from coralsep.data.prepare.but_reverbdb import (
    _SLR17_ARCHIVES,
    _SLR17_BASE_URL,
    _find_rir_wavs,
)


def test_download_url_points_at_but_not_openslr():
    assert "openslr.org" not in _SLR17_BASE_URL
    assert _SLR17_BASE_URL == "http://merlin.fit.vutbr.cz/ReverbDB/"


def test_archive_filename_matches_the_real_published_name():
    """The real archive is named ...RIR-Only.tgz. The old code was missing
    the "-Only" suffix, which alone would have 404'd even on the right host."""
    filenames = [name for name, _ in _SLR17_ARCHIVES]
    assert "BUT_ReverbDB_rel_19_06_RIR-Only.tgz" in filenames


def test_every_archive_url_is_built_from_the_real_base():
    for _, url in _SLR17_ARCHIVES:
        assert url.startswith("http://merlin.fit.vutbr.cz/ReverbDB/")


def test_find_rir_wavs_excludes_silence_recordings(tmp_path):
    """
    Regression for a second I-053 defect found while running the corrected
    download for real: the archive lays each session out as sibling RIR/
    and silence/ directories. RIR/ holds a real, short impulse response;
    silence/ holds a 60-second background noise recording with no impulse
    at all. Confirmed against a real download that measuring T60 on the
    silence file (which the old unfiltered rglob did) produces a
    physically impossible T60 in the tens to hundreds of seconds.
    """
    session = tmp_path / "RoomA" / "Mic01" / "Session1" / "17"
    (session / "RIR").mkdir(parents=True)
    (session / "silence").mkdir(parents=True)
    real_rir = session / "RIR" / "IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav"
    real_rir.write_bytes(b"not real audio, path filtering is what is tested here")
    noise = session / "silence" / "silence_16kHz_60sec.v00.wav"
    noise.write_bytes(b"not real audio, path filtering is what is tested here")

    found = _find_rir_wavs(tmp_path)

    assert real_rir in [f.resolve() for f in found] or real_rir.resolve() in [
        f.resolve() for f in found
    ]
    assert noise.resolve() not in [f.resolve() for f in found]


def test_find_rir_wavs_still_works_on_a_flat_manual_fallback_directory(tmp_path):
    """The documented manual-placement fallback (files dropped directly under
    the extracted directory, no RIR/ or silence/ structure) must still find
    them: the fix excludes "silence" paths, it does not require "RIR"."""
    flat_rir = tmp_path / "my_manually_placed_rir.wav"
    flat_rir.write_bytes(b"not real audio")

    found = _find_rir_wavs(tmp_path)

    assert [f.resolve() for f in found] == [flat_rir.resolve()]
