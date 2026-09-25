"""Single shared implementation of the benchmark's change detection metrics."""

from .bootstrap import bootstrap_ci, paired_bootstrap
from .labels import SECOND_CLASSES, SECOND_COLORMAP, index_to_rgb, rgb_to_index, to_index
from .scd import (
    BCDMeter,
    SCDMeter,
    binary_from_semantic,
    binary_scores,
    cohen_kappa,
    compose_prediction,
    confusion,
    fromto_map,
    scd_scores,
)

__all__ = [
    "BCDMeter", "SCDMeter", "binary_from_semantic", "binary_scores", "bootstrap_ci",
    "cohen_kappa", "compose_prediction", "confusion", "index_to_rgb", "paired_bootstrap",
    "fromto_map", "rgb_to_index", "scd_scores", "to_index", "SECOND_CLASSES", "SECOND_COLORMAP",
]
