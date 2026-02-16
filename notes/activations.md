# Activation Functions

## Classic Activations (Review)
- **ReLU**: max(0, x) — most common, simple, effective
- **Sigmoid**: 1/(1+e^(-x)) — output in (0,1), used for binary classification
- **Tanh**: (e^x - e^(-x))/(e^x + e^(-x)) — output in (-1,1), zero-centered

---

## Modern Activations

### GELU (Gaussian Error Linear Unit)
- **Formula**: x · Φ(x), where Φ is the CDF of standard normal distribution
- **Approximation**: 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
- **Intuition**: Stochastically multiply input by 0 or 1 based on how much greater it is than other inputs
- **Properties**:
  - **Smooth** (unlike ReLU which has a kink at 0)
  - Non-monotonic (has a slight negative region)
  - Used in BERT, GPT, many transformers
- **Why it works**: Combines properties of dropout and ReLU; probabilistic interpretation

### Swish (also called SiLU - Sigmoid Linear Unit)
- **Formula**: x · σ(βx), where σ is sigmoid
- **When β=1**: x · sigmoid(x)
- **Properties**:
  - Smooth, non-monotonic
  - Self-gated (input gates itself)
  - Discovered by Google via neural architecture search
- **Performance**: Often outperforms ReLU on deeper networks

### Mish
- **Formula**: x · tanh(softplus(x)) = x · tanh(ln(1 + e^x))
- **Properties**:
  - Smooth, non-monotonic
  - Unbounded above, bounded below
  - Similar benefits to Swish but sometimes better
- **Use case**: State-of-the-art in some vision tasks (YOLOv4 uses it)

---

## S3/S4 Activation Functions

**Note**: S3 and S4 are less common in mainstream literature. Context suggests:

### Possible Interpretations:
1. **Structured State Space Sequence (S4) Models**:
   - Not an activation function, but a model architecture
   - Alternative to transformers for sequence modeling
   - Uses linear state-space models
   - Efficient for very long sequences

2. **Sigmoid-Weighted Linear Units (SiLU/Swish family)**:
   - S3/S4 might refer to variations in the Swish family
   - Different values of β parameter

**Clarification needed**: Check with professor or course materials for specific definition in your course context.

---

## Choosing Activation Functions

### General Guidelines:
- **Default choice**: ReLU (or variants like Leaky ReLU)
- **Transformers/NLP**: GELU is standard
- **Deep networks**: Swish or Mish may help
- **Binary output**: Sigmoid
- **Multi-class output**: Softmax
- **Output layer for regression**: Linear (no activation)

### Why newer activations work:
1. **Smoothness**: Helps with optimization (better gradients)
2. **Non-monotonicity**: Can model more complex functions
3. **Self-gating**: Input modulates its own flow
