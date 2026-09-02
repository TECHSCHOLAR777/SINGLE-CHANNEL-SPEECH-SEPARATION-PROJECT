"""CoRAL-Sep: Condition-Routed Adapter Library for speech separation.

A frozen pretrained separation backbone, left untouched, steered by three small
LoRA adapters that are blended into its weights in proportion to the reverb,
noise and codec damage measured in the input.
"""

__version__ = "0.2.0"
