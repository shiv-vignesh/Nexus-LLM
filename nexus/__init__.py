"""
Nexus-LLM: End-to-end LLM post-training, evaluation, and production serving platform.

This package provides:
- Training: RLHF reward model training with LoRA fine-tuning
- Registry: Model versioning and artifact management
- Inference: vLLM-backed high-performance inference engine
- Serving: FastAPI-based production serving with Redis queuing
- Evaluation: Comprehensive metrics for generation and alignment quality
"""

__version__ = "0.1.0"
__author__ = "Shiv Vignesh"

from nexus.core.config import NexusConfig
from nexus.core.exceptions import NexusError

__all__ = ["NexusConfig", "NexusError", "__version__"]
