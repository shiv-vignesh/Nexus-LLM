# Nexus-LLM

**End-to-End LLM Post-Training, Evaluation, and Production Serving Platform**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Nexus-LLM is a comprehensive platform that unifies the entire LLM lifecycle: from RLHF post-training to high-performance inference serving. 

## Key Features

| Component | Capabilities |
|-----------|-------------|
| **Training** | RLHF reward model training, LoRA fine-tuning, pairwise preference learning |
| **Registry** | Model versioning, artifact storage, stage-based lifecycle management |
| **Inference** | vLLM high-performance serving, prefix caching, continuous batching |
| **Serving** | OpenAI-compatible API, Redis-backed queuing, horizontal scaling |
| **Evaluation** | BLEU/ROUGE metrics, reward accuracy, latency benchmarking |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NEXUS-LLM PLATFORM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   TRAINING   │───▶│    MODEL     │───▶│  INFERENCE   │                  │
│  │   PIPELINE   │    │   REGISTRY   │    │    ENGINE    │                  │
│  │              │    │              │    │              │                  │
│  │  • RLHF RM   │    │  • Versions  │    │  • vLLM      │                  │
│  │  • LoRA      │    │  • Metadata  │    │  • Batching  │                  │
│  │  • Pairwise  │    │  • Stages    │    │  • Caching   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                   │                          │
│                                                   ▼                          │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                       SERVING LAYER (FastAPI)                     │      │
│  │  ┌──────────────────────────────────────────────────────────┐    │      │
│  │  │ POST /v1/completions    POST /v1/chat/completions        │    │      │
│  │  │ GET  /v1/models         GET  /health                     │    │      │
│  │  └──────────────────────────────────────────────────────────┘    │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                     INFRASTRUCTURE LAYER                          │      │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │      │
│  │   │  Redis  │    │ Dynamic │    │ Prefix  │    │  Model  │      │      │
│  │   │ Streams │    │ Batcher │    │  Cache  │    │ Storage │      │      │
│  │   └─────────┘    └─────────┘    └─────────┘    └─────────┘      │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nexus-llm.git
cd nexus-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e ".[dev]"

# For vLLM support (requires CUDA)
pip install vllm
```

## Quick Start

### 1. Train a Reward Model

```python
from nexus.training import RewardModelTrainer
from nexus.training.data import RLHFDataset
from nexus.core.config import TrainingConfig

# Configure training
config = TrainingConfig(
    base_model="facebook/opt-1.3b",
    use_lora=True,
    learning_rate=1e-5,
    num_epochs=3,
)

# Load preference data
dataset = RLHFDataset("data/preferences.jsonl")

# Train
trainer = RewardModelTrainer(config)
result = trainer.train(dataset, output_dir="./output")
print(f"Best loss: {result.best_loss:.4f}")
```

Or use the CLI:
```bash
python scripts/train_reward_model.py \
    --base-model facebook/opt-1.3b \
    --data-path data/train.jsonl \
    --epochs 3 \
    --lora \
    --register
```

### 2. Register the Model

```python
from nexus.registry import ModelRegistry, ModelType

registry = ModelRegistry()

version = registry.register_model(
    model_id="my-reward-model",
    model_type=ModelType.REWARD_MODEL,
    artifact_path="./output/final_model",
    base_model="facebook/opt-1.3b",
    training_metrics={"accuracy": 0.85, "loss": 0.32},
)

# Promote to production
registry.promote_version("my-reward-model", version.version, ModelStage.PRODUCTION)
```

### 3. Serve with vLLM

```python
from nexus.inference import VLLMInferenceEngine
from nexus.core.config import InferenceConfig
from nexus.core.interfaces import GenerationRequest

# Initialize engine
config = InferenceConfig(
    model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    enable_prefix_caching=True,
)
engine = VLLMInferenceEngine(config)
await engine.initialize()

# Generate
request = GenerationRequest(
    request_id="1",
    prompt="Explain quantum computing:",
    max_tokens=256,
    temperature=0.7,
)
response = await engine.generate(request)
print(response.generated_text)
```

### 4. Deploy API Server

```bash
# Start with uvicorn
python -m nexus.serving.app

# Or with Docker Compose
cd docker && docker-compose up -d
```

Test the API:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### 5. Benchmark Performance

```bash
python scripts/benchmark_inference.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --backend vllm \
    --requests 100 \
    --concurrency 10
```

## Project Structure

```
nexus-llm/
├── nexus/
│   ├── core/                    # Core abstractions & configuration
│   │   ├── config.py            # Pydantic settings management
│   │   ├── exceptions.py        # Custom exception hierarchy
│   │   └── interfaces/          # Abstract protocols (ITrainer, IInferenceEngine)
│   │
│   ├── training/                # RLHF Training Pipeline
│   │   ├── reward_model/        # Reward model trainer, loss functions
│   │   ├── lora/                # LoRA configuration factory
│   │   ├── data/                # Dataset, collator, preprocessing
│   │   └── callbacks/           # Checkpointing, early stopping
│   │
│   ├── registry/                # Model Registry & Versioning
│   │   ├── registry.py          # ModelRegistry implementation
│   │   ├── storage/             # Local and S3 storage backends
│   │   └── metadata.py          # Model card and artifact schemas
│   │
│   ├── inference/               # High-Performance Inference
│   │   ├── engines/             # vLLM and Transformers backends
│   │   ├── batching/            # Dynamic request batching
│   │   ├── caching/             # Prefix cache management
│   │   └── generation/          # Sampling configuration
│   │
│   ├── evaluation/              # Evaluation & Metrics
│   │   ├── metrics/             # BLEU, ROUGE, latency metrics
│   │   └── benchmarks/          # Throughput benchmarking
│   │
│   ├── serving/                 # FastAPI Serving Layer
│   │   ├── app.py               # Application factory
│   │   ├── routers/             # API endpoints (OpenAI-compatible)
│   │   ├── middleware/          # Rate limiting, logging
│   │   └── schemas/             # Pydantic request/response models
│   │
│   ├── workers/                 # Background Workers
│   │   └── inference_worker.py  # Redis stream consumer
│   │
│   └── infrastructure/          # Infrastructure Adapters
│       ├── redis/               # Redis client, streams, cache
│       └── logging/             # Structured logging
│
├── scripts/                     # CLI tools
├── configs/                     # YAML configuration files
├── docker/                      # Docker and docker-compose
└── tests/                       # Test suite
```

## Design Principles

### Clean Architecture
- **Domain-driven design** with clear separation between core logic and infrastructure
- **Dependency injection** via FastAPI's Depends and manual constructor injection
- **Interface abstractions** (ITrainer, IInferenceEngine, IModelRegistry) for testability

### SOLID Principles
- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Extend via new engine implementations, not modifications
- **Liskov Substitution**: vLLM and Transformers engines are interchangeable
- **Interface Segregation**: Minimal interfaces focused on specific capabilities
- **Dependency Inversion**: High-level modules depend on abstractions

### Production Patterns
- **Token bucket rate limiting** for API protection
- **Redis Streams** for reliable distributed task queuing
- **Dynamic batching** with size and timeout triggers
- **Structured logging** with JSON output for observability
- **Graceful shutdown** handling for zero-downtime deploys

## Configuration

Environment variables (with `NEXUS_` prefix):
```bash
# Inference
NEXUS_SERVING__INFERENCE__MODEL_NAME_OR_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
NEXUS_SERVING__INFERENCE__BACKEND=vllm

# Redis
NEXUS_SERVING__REDIS__HOST=localhost
NEXUS_SERVING__REDIS__PORT=6379

# Logging
NEXUS_LOG_LEVEL=INFO
```

Or use YAML configs:
```yaml
# configs/serving/production.yaml
serving:
  inference:
    backend: "vllm"
    model_name_or_path: "meta-llama/Llama-2-7b-hf"
    enable_prefix_caching: true
    gpu_memory_utilization: 0.85
```

## API Reference

### Completions (OpenAI-compatible)

```bash
POST /v1/completions
{
  "prompt": "Hello, how are",
  "max_tokens": 50,
  "temperature": 0.7
}
```

### Chat Completions

```bash
POST /v1/chat/completions
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi!"}
  ],
  "max_tokens": 100,
  "stream": true
}
```

### Health Check

```bash
GET /health
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600
}
```

## Deployment

### Docker Compose (Recommended)

```bash
cd docker
docker-compose up -d

# Scale workers
docker-compose up -d --scale worker=4
```

### Kubernetes

See `k8s/` directory for Helm charts and manifests (coming soon).

## Performance

Benchmarks on NVIDIA A100 (40GB):

| Model | Backend | Throughput | P50 Latency | P99 Latency |
|-------|---------|------------|-------------|-------------|
| Llama-2-7B | vLLM | 180 tok/s | 45ms | 120ms |
| Llama-2-7B | Transformers | 35 tok/s | 180ms | 450ms |
| TinyLlama-1.1B | vLLM | 850 tok/s | 12ms | 35ms |

*Prefix caching enabled, batch size 32, 128 output tokens*

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black nexus/
ruff check nexus/

# Type checking
mypy nexus/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) for high-performance inference
- [HuggingFace Transformers](https://github.com/huggingface/transformers) for model loading
- [PEFT](https://github.com/huggingface/peft) for LoRA implementation
- [TRL](https://github.com/huggingface/trl) for RLHF utilities
