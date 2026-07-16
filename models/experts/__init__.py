"""Frozen pretrained separation backbone wrappers.

CALM-Sep uses a single frozen SR-CorrNet var-2-5 backbone. The old
multi-expert bank (MossFormer2, SepFormer, TF-GridNet) was removed in the
architecture pivot; see BLUEPRINT §3 and the README migration table.
"""

from models.experts.embeddings import ECAPAEmbedder
from models.experts.srcorrnet import SRCorrNetExpert

__all__ = [
    "ECAPAEmbedder",
    "SRCorrNetExpert",
]
