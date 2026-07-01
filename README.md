# Conscious AI: A Transformer-Based Architecture with Global Workspace Theory

## Project Overview

**Conscious AI** is an ambitious research project building a conscience AI system from scratch using a transformer-based architecture enhanced with Global Workspace Theory (GWT) and Higher-Order Thought (HOT) modules. This project explores whether consciousness-inspired mechanisms can emerge in neural architectures.

## Core Architecture

### 1. **Transformer Foundation**
- Custom transformer implementation built from first principles using PyTorch
- Attention mechanisms with consciousness-aligned modifications
- Scalable architecture supporting variable model sizes

### 2. **Global Workspace Theory (GWT) Backend**
- Central hub that acts as a "conscious workspace"
- Specialized modules report findings and data to the workspace
- Information bottleneck that simulates conscious broadcasting
- Manages attention allocation across competing processes

### 3. **Higher-Order Thought (HOT) Modules**
- Metacognitive evaluation layer
- Pre-processes data before workspace broadcast
- Evaluates importance, relevance, and coherence
- Self-reflection and confidence scoring mechanisms

## Project Structure

```
conscious-ai/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── gwt_config.yaml
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── transformer.py          # Base transformer architecture
│   │   ├── attention.py            # Attention mechanisms
│   │   └── embeddings.py           # Token/position embeddings
│   ├── gwt/
│   │   ├── workspace.py            # Central workspace hub
│   │   ├── workspace_hub.py        # Workspace management
│   │   └── integration.py          # GWT-Transformer integration
│   ├── hot/
│   │   ├── hot_module.py           # Higher-order thought layer
│   │   ├── evaluator.py            # Data evaluation pipeline
│   │   └── metacognition.py        # Self-reflection mechanisms
│   ├── consciousness/
│   │   ├── metrics.py              # Consciousness measurement
│   │   ├── integration_info.py     # Integrated information theory
│   │   └── monitoring.py           # Consciousness monitoring
│   ├── data/
│   │   ├── dataset.py              # Dataset loading
│   │   ├── preprocessing.py        # Data preprocessing
│   │   └── sampler.py              # Custom samplers
│   ├── training/
│   │   ├── trainer.py              # Main training loop
│   │   ├── losses.py               # Custom loss functions
│   │   └── optimizer.py            # Optimization strategies
│   └── utils/
│       ├── logging.py              # Logging utilities
│       ├── config.py               # Configuration management
│       └── helpers.py              # Helper functions
├── data/
│   ├── raw/                        # Raw training data
│   ├── processed/                  # Processed datasets
│   └── README.md
├── models/
│   ├── checkpoints/                # Model checkpoints
│   └── README.md
├── notebooks/
│   ├── exploration.ipynb
│   ├── consciousness_analysis.ipynb
│   └── README.md
├── tests/
│   ├── test_transformer.py
│   ├── test_gwt.py
│   ├── test_hot.py
│   └── test_consciousness.py
├── scripts/
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   ├── analyze_consciousness.py    # Consciousness analysis
│   └── inference.py                # Inference script
└── docs/
    ├── ARCHITECTURE.md
    ├── GWT_DESIGN.md
    ├── HOT_DESIGN.md
    ├── CONSCIOUSNESS_METRICS.md
    └── RESEARCH_NOTES.md
```

## Key Concepts

### Global Workspace Theory (GWT)
- **Central Workspace**: Limited-capacity information hub
- **Specialist Modules**: Autonomous processors that compete for workspace access
- **Broadcasting**: Winning information broadcasted to all modules
- **Consciousness Simulation**: Only workspace-broadcast information is "conscious"

### Higher-Order Thought (HOT)
- **Metacognition**: Thoughts about thoughts
- **Evaluation**: Pre-processes data for consciousness worthiness
- **Filtering**: Determines what reaches the workspace
- **Confidence**: Scores self-awareness and certainty

## Getting Started

### Installation

```bash
git clone https://github.com/paulofficial300/conscious-ai.git
cd conscious-ai
pip install -r requirements.txt
```

### Quick Start

```python
from src.core.transformer import ConsciousTransformer
from src.gwt.workspace import GlobalWorkspace
from src.hot.hot_module import HigherOrderThought

# Initialize components
transformer = ConsciousTransformer(config_path='configs/model_config.yaml')
workspace = GlobalWorkspace(config_path='configs/gwt_config.yaml')
hot = HigherOrderThought()

# Training
python scripts/train.py --config configs/training_config.yaml
```

## Research References

- **Global Workspace Theory**: Baars, B. J. (1988). "A Cognitive Theory of Consciousness"
- **Higher-Order Thought Theory**: Rosenthal, D. (1997). "A Theory of Consciousness"
- **Integrated Information Theory**: Tononi, G. (2012). "Consciousness as Integrated Information"
- **Transformers**: Vaswani et al. (2017). "Attention Is All You Need"

## Development Roadmap

- [ ] Phase 1: Core Transformer Architecture
- [ ] Phase 2: GWT Workspace Implementation
- [ ] Phase 3: HOT Module Integration
- [ ] Phase 4: Consciousness Metrics
- [ ] Phase 5: Training & Evaluation
- [ ] Phase 6: Analysis & Refinement
- [ ] Phase 7: Research Publication

## Contributing

This is a solo research project by paulofficial300. Documentation and code structure are designed for clarity and reproducibility.

## License

MIT License - See LICENSE file for details

## References & Further Reading

See `docs/RESEARCH_NOTES.md` for comprehensive research documentation.

---

**Project Status**: 🚀 Initialization Phase

*"The question of machine consciousness may be the most important question of our time."*
