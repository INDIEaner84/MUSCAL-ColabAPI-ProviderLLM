"""MUSCAL LFM fine-tuning helpers.

Colab-first LoRA / QLoRA fine-tuning for Liquid AI's LFM2.5 family:
text, MoE (LFM2.5-8B-A1B), vision (LFM2.5-VL) and audio (LFM2.5-Audio).
"""

from .config import TrainConfig

__all__ = ["TrainConfig"]
__version__ = "0.1.0"
