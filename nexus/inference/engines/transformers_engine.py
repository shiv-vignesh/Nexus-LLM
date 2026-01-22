"""
HuggingFace Transformers inference engine.

Provides a fallback inference engine when vLLM is not available
or for debugging/development purposes.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextIteratorStreamer,
)

from nexus.core.config import InferenceConfig
from nexus.core.interfaces.inference_engine import (
    GenerationRequest,
    GenerationResponse,
    StreamingChunk,
)
from nexus.core.exceptions import InferenceError, EngineNotReadyError, ModelLoadError
from nexus.inference.engines.base import BaseInferenceEngine


class TransformersInferenceEngine(BaseInferenceEngine):
    """HuggingFace Transformers-based inference engine.

    A flexible inference engine using HuggingFace Transformers.
    Less optimized than vLLM but more portable and easier to debug.

    Features:
    - Support for any HuggingFace model
    - PEFT/LoRA adapter loading
    - Mixed precision inference
    - Thread pool for async execution

    Example:
        >>> config = InferenceConfig(
        ...     model_name_or_path="facebook/opt-1.3b",
        ...     backend=InferenceBackend.TRANSFORMERS
        ... )
        >>> engine = TransformersInferenceEngine(config)
        >>> await engine.initialize()
        >>> response = await engine.generate(request)
    """

    def __init__(
        self,
        config: InferenceConfig,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
    ):
        """Initialize Transformers engine.

        Args:
            config: Inference configuration
            model: Optional pre-loaded model
            tokenizer: Optional pre-loaded tokenizer
        """
        super().__init__(config)
        self._model = model
        self._tokenizer = tokenizer
        self._device: torch.device | None = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._model_info: dict = {}

    async def initialize(self) -> None:
        """Initialize model and tokenizer.

        Loads the model and moves it to the appropriate device.
        """
        try:
            # Determine device
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")

            # Load tokenizer
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name_or_path,
                    trust_remote_code=True,
                )

                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model
            if self._model is None:
                dtype = torch.float16 if self._device.type == "cuda" else torch.float32

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name_or_path,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    device_map="auto" if self._device.type == "cuda" else None,
                )

                if self._device.type != "cuda":
                    self._model = self._model.to(self._device)

            self._model.eval()

            # Store model info
            self._model_info = {
                "model_name": self.config.model_name_or_path,
                "device": str(self._device),
                "dtype": str(self._model.dtype),
                "num_parameters": sum(p.numel() for p in self._model.parameters()),
                "backend": "transformers",
            }

            self._is_ready = True
            print(f"Transformers engine initialized with {self.config.model_name_or_path}")

        except Exception as e:
            raise ModelLoadError(self.config.model_name_or_path, str(e))

    def _generate_sync(
        self,
        prompt: str,
        sampling_config: dict,
    ) -> tuple[str, int, int, str]:
        """Synchronous generation (runs in thread pool).

        Returns:
            Tuple of (generated_text, prompt_tokens, completion_tokens, finish_reason)
        """
        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_model_len,
        ).to(self._device)

        prompt_tokens = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                **sampling_config,
            )

        # Decode (only the generated part)
        generated_ids = outputs[0][prompt_tokens:]
        generated_text = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        completion_tokens = len(generated_ids)

        # Determine finish reason
        if self._tokenizer.eos_token_id in generated_ids:
            finish_reason = "stop"
        elif completion_tokens >= sampling_config.get("max_new_tokens", 256):
            finish_reason = "length"
        else:
            finish_reason = "stop"

        return generated_text, prompt_tokens, completion_tokens, finish_reason

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Transformers.

        Args:
            request: Generation request

        Returns:
            GenerationResponse with generated text
        """
        if not self._is_ready or self._model is None:
            raise EngineNotReadyError("transformers")

        start_time = time.perf_counter()

        try:
            # Create HF generation config
            sampling_config = self._create_sampling_config(request)
            hf_config = sampling_config.to_hf_config()

            # Add stop sequences if provided
            if request.stop_sequences:
                stop_token_ids = []
                for seq in request.stop_sequences:
                    tokens = self._tokenizer.encode(seq, add_special_tokens=False)
                    if tokens:
                        stop_token_ids.append(tokens[0])
                if stop_token_ids:
                    hf_config["eos_token_id"] = stop_token_ids

            # Run generation in thread pool
            loop = asyncio.get_event_loop()
            generated_text, prompt_tokens, completion_tokens, finish_reason = await loop.run_in_executor(
                self._executor,
                self._generate_sync,
                request.prompt,
                hf_config,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_metrics(completion_tokens, latency_ms)

            return self._create_response(
                request=request,
                generated_text=generated_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
            )

        except Exception as e:
            raise InferenceError(f"Generation failed: {e}")

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamingChunk]:
        """Stream generation using TextIteratorStreamer.

        Args:
            request: Generation request

        Yields:
            StreamingChunk objects as tokens are generated
        """
        if not self._is_ready or self._model is None:
            raise EngineNotReadyError("transformers")

        try:
            # Tokenize
            inputs = self._tokenizer(
                request.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_model_len,
            ).to(self._device)

            # Create streamer
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            # Create generation config
            sampling_config = self._create_sampling_config(request)
            hf_config = sampling_config.to_hf_config()

            # Start generation in separate thread
            generation_kwargs = {
                **inputs,
                **hf_config,
                "streamer": streamer,
                "pad_token_id": self._tokenizer.pad_token_id,
            }

            loop = asyncio.get_event_loop()
            generation_task = loop.run_in_executor(
                self._executor,
                lambda: self._model.generate(**generation_kwargs),
            )

            # Stream tokens
            full_text = ""
            try:
                for text in streamer:
                    full_text += text
                    yield StreamingChunk(
                        request_id=request.request_id,
                        text=text,
                        is_final=False,
                    )

                # Final chunk
                yield StreamingChunk(
                    request_id=request.request_id,
                    text="",
                    is_final=True,
                    finish_reason="stop",
                )

            finally:
                # Ensure generation completes
                await generation_task

        except Exception as e:
            yield StreamingChunk(
                request_id=request.request_id,
                text="",
                is_final=True,
                finish_reason="error",
            )
            raise InferenceError(f"Streaming generation failed: {e}")

    async def health_check(self) -> bool:
        """Check if engine is healthy."""
        if not self._is_ready or self._model is None:
            return False

        try:
            # Quick forward pass check
            test_input = self._tokenizer("test", return_tensors="pt").to(self._device)
            with torch.no_grad():
                _ = self._model(**test_input)
            return True
        except Exception:
            return False

    async def get_model_info(self) -> dict:
        """Get model information."""
        return self._model_info.copy()

    async def shutdown(self) -> None:
        """Shutdown engine and release resources."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            self._tokenizer = None

        self._executor.shutdown(wait=False)
        self._is_ready = False

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Transformers engine shutdown complete")

    @property
    def backend_name(self) -> str:
        """Get backend name."""
        return "transformers"

    def load_lora_adapter(self, adapter_path: str) -> None:
        """Load a LoRA adapter onto the model.

        Args:
            adapter_path: Path to the PEFT adapter
        """
        if self._model is None:
            raise EngineNotReadyError("transformers")

        try:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(
                self._model,
                adapter_path,
            )
            self._model.eval()

            self._model_info["lora_adapter"] = adapter_path
            print(f"Loaded LoRA adapter from {adapter_path}")

        except ImportError:
            raise ImportError("PEFT is required to load LoRA adapters")
        except Exception as e:
            raise ModelLoadError(adapter_path, f"Failed to load adapter: {e}")
