"""Compatibility tests for the GeomorphConn namespace."""

import geomorphconn


def test_geomorphconn_imports_connectivity_index():
    assert hasattr(geomorphconn, "ConnectivityIndex")


def test_geomorphconn_submodule_aliases():
    from geomorphconn.components import ConnectivityIndex as CI2

    assert CI2 is geomorphconn.ConnectivityIndex
