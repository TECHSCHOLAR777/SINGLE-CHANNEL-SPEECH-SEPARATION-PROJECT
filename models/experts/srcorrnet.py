"""CALM-Sep: re-export backbone wrapper under the models.experts namespace.

The canonical implementation lives at models.srcorrnet. This module keeps
models.experts importable with the SRCorrNetExpert name.
"""

from models.srcorrnet import SRCorrNetWrapper as SRCorrNetExpert

__all__ = ["SRCorrNetExpert"]
