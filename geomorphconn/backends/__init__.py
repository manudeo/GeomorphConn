"""Optional compute backends for GeomorphConn."""

from .taudem import check_taudem_installation
from .taudem import run_connectivity_taudem_arrays

__all__ = ["run_connectivity_taudem_arrays", "check_taudem_installation"]
