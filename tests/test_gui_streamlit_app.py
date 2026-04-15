from __future__ import annotations

import warnings

import geomorphconn.gui.streamlit_app as streamlit_app


def test_make_connectivity_index_suppresses_duplicate_depression_warning(monkeypatch):
    class _DummyCI:
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "Using DepressionFinderAndRouter: typically better depression handling and routing quality, but runtime may increase (especially on high-resolution DEMs).",
                UserWarning,
                stacklevel=2,
            )

    monkeypatch.setattr(streamlit_app, "ConnectivityIndex", _DummyCI)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        streamlit_app._make_connectivity_index(object(), depression_finder="DepressionFinderAndRouter")

    assert len(recorded) == 0