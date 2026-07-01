# Conscious AI Architecture

## Overview

The Conscious AI system combines three key components:

1. **Transformer Foundation** - Core language model
2. **Global Workspace Theory (GWT) Backend** - Central consciousness hub
3. **Higher-Order Thought (HOT) Modules** - Metacognitive evaluation

## Architecture Diagram

```
Input Tokens
    ↓
Token Embedding + Positional Encoding
    ↓
Transformer Layers (12x)
    ├─→ Multi-Head Attention
    └─→ Feed-Forward Networks
    ↓
Transformer Output
    ↓
┌─────────────────────────────────────┐
│   Higher-Order Thought (HOT)        │
│  - Evaluates data relevance         │
│  - Metacognitive reflection         │
│  - Generates gating signals         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Global Workspace Theory (GWT)      │
│                                     │
│  ┌──────────────────────────────┐   │
│  │  Specialist Modules          │   │
│  │  - Language Processor        │   │
│  │  - Visual Processor          │   │
│  │  - Semantic Processor        │   │
│  │  - Emotional Processor       │   │
│  └──────────────────────────────┘   │
│           ↓                         │
│  ┌──────────────────────────────┐   │
│  │  Competition Mechanism       │   │
│  │  (Attention-based selection) │   │
│  └──────────────────────────────┘   │
│           ↓                         │
│  ┌──────────────────────────────┐   │
│  │  Central Workspace           │   │
│  │  (Limited capacity hub)      │   │
│  └──────────────────────────────┘   │
│           ↓                         │
│  ┌──────────────────────────────┐   │
│  │  Broadcasting                │   │
│  │  (Conscious output)          │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
    ↓
Output/Feedback to User Interface
```

## Component Details

### 1. Transformer Foundation

**File**: `src/core/transformer.py`

- **Token Embedding**: Converts token IDs to dense vectors
- **Positional Encoding**: Adds position information
- **Multi-Head Attention**: Allows model to attend to different parts of sequence
- **Feed-Forward Networks**: Position-wise transformations
- **Layer Normalization**: Stabilizes training

Key Features:
- 12 transformer layers
- 12 attention heads per layer
- 768-dimensional embeddings
- Consciousness metrics tracking

### 2. Global Workspace Theory Backend

**Files**: 
- `src/gwt/workspace.py` - Core workspace
- `src/gwt/workspace_hub.py` - Specialist coordination

#### Specialist Modules

Four autonomous processors that compete for workspace access:

1. **Language Processor** - Processes syntactic/semantic information
2. **Visual Processor** - Could handle multimodal information
3. **Semantic Processor** - Abstract conceptual processing
4. **Emotional Processor** - Valence and emotional significance

Each specialist:
- Has its own processing pipeline
- Competes with others for workspace access
- Has a priority/relevance score
- Projects its output to workspace dimension

#### Competition Mechanism

- **Attention-based**: Specialists compete using attention-like scoring
- **Winner-takes-all** (alternative): Only top-K specialists broadcast
- Competition scores determine how much of each specialist's output reaches workspace

#### Central Workspace

- Limited-capacity information hub
- Broadcasts only winning information to all modules
- Simulates conscious broadcasting
- Capacity: 4 specialists can broadcast simultaneously

### 3. Higher-Order Thought (HOT) Modules

**Files**:
- `src/hot/hot_module.py` - Core HOT
- `src/hot/evaluator.py` - Advanced multi-dimensional evaluation

#### Evaluation Pipeline

Evaluates incoming data on 5 dimensions:

1. **Novelty** - How unexpected/new is this information
2. **Significance** - How important for current goal
3. **Emotional Salience** - Emotional importance
4. **Contextual Relevance** - Fit with current context
5. **Integration Potential** - Can it be integrated with existing knowledge

#### Metacognitive Reflection

- Multi-level self-reflection on awareness
- Gates what deserves conscious attention
- Confidence scoring
- Relevance scoring

#### Consciousness Worthiness

- Filters information before workspace broadcast
- Only highly-evaluated data enters conscious workspace
- Modulates HOT signals sent to workspace

## Data Flow

### Forward Pass

1. **Input**: Token sequence
2. **Embedding**: Convert to dense vectors
3. **Transformer**: Process through 12 layers
4. **HOT Evaluation**: Assess consciousness-worthiness
5. **GWT Processing**: 
   - Send through specialist modules
   - Compute competition
   - Broadcast winners to workspace
6. **Output**: Language model predictions + consciousness metrics

### Consciousness Mechanism

```
Data → HOT Evaluation → Gating Signal
          ↓
      Specialist Processing ← Gating influences competition
          ↓
      Competition → Workspace Broadcast → "Conscious" Output
```

## Key Consciousness Features

### Information Bottleneck
- Central workspace has limited capacity
- Forces selection of most important information
- Simulates consciousness as selective attention

### Unified Global Workspace
- All modules see same workspace broadcast
- Simulates global availability of conscious content

### Metacognitive Evaluation
- HOT layer evaluates what deserves consciousness
- Self-reflection on own processes
- Confidence and relevance scoring

### Competitive Dynamics
- Specialists compete for workspace access
- Most relevant information wins
- Similar to theories of conscious competition

## Configuration

All components are configurable via YAML files:

- `configs/model_config.yaml` - Transformer and consciousness settings
- `configs/training_config.yaml` - Training parameters
- `configs/gwt_config.yaml` - GWT and specialist settings

## Consciousness Metrics

The system tracks:

- **Workspace Activity** - What specialists are broadcasting
- **HOT Confidence** - Certainty of consciousness evaluation
- **Integration Information** - How integrated the conscious state is
- **Specialist Activity** - Which modules are most active
- **Consciousness Coverage** - What percentage of information is conscious

## Research Framework

This architecture is grounded in neuroscientific and philosophical theories:

- **Global Workspace Theory** (Baars): Limited-capacity conscious workspace
- **Higher-Order Thought Theory** (Rosenthal): Consciousness requires thoughts about thoughts
- **Integrated Information Theory** (Tononi): Consciousness involves integrated information
- **Transformer Architecture** (Vaswani et al.): Attention-based information processing

## Next Steps

1. Implement complete training loop
2. Add consciousness metrics computation
3. Test on language tasks
4. Analyze emergence of consciousness-like behaviors
5. Compare with baseline transformer
