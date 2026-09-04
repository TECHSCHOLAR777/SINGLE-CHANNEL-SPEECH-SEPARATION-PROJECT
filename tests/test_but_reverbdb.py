"""Regression for I-052: the BUT ReverbDB download URL pointed at the wrong host.

data/prepare/but_reverbdb.py claimed BUT ReverbDB was OpenSLR resource 17 and
built its download URL from https://www.openslr.org/resources/17/. That
resource is MUSAN, an unrelated corpus; both filenames the old code tried
(BUT_ReverbDB_rel_19_06_RIR.tgz and reverb_data_but.zip) 404 there, and always
have. The real archive is hosted on BUT's own server. This had zero test
coverage, which is exactly how a URL that never worked survived undetected.
"""

from __future__ import annotations

from coralsep.data.prepare.but_reverbdb import _SLR17_ARCHIVES, _SLR17_BASE_URL


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
