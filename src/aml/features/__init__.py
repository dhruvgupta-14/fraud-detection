"""Feature blocks. Importing this package registers every block.

The registry is populated as a side effect of import, so anything calling
``enabled_blocks(cfg)`` must have imported this package first -- importing it here means no
caller has to think about it.
"""

from aml.features import motifs, streaming, structural, tabular  # noqa: F401  (registration)
from aml.features.base import (  # noqa: F401
    Causality,
    FeatureBlock,
    FeatureContext,
    FeatureSpec,
    build_manifest,
    enabled_blocks,
    get_block,
    register,
)
