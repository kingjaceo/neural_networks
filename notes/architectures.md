# Neural Network Architectures

## ResNet (Residual Networks)

### The Problem ResNet Solves
- Deeper networks should be at least as good as shallower ones
- **In practice**: Very deep networks (50+ layers) train *worse* than shallower ones
- **Not** overfitting — training error actually gets worse!
- **Reason**: Vanishing gradients and degradation problem

### The Solution: Skip Connections

#### Standard Layer:
```
output = ReLU(Conv(input))
```

#### Residual Block:
```
output = ReLU(Conv(input) + input)
                           ↑
                    skip connection
```

### Why Connecting Input to Output ReLU is a Good Idea

#### 1. **Gradient Flow**
- Gradients can flow **directly back** through the skip connection
- Avoids vanishing gradient problem
- Even in a 152-layer network, gradients reach early layers

#### 2. **Easier Optimization**
- Network learns **residual mapping** F(x) instead of H(x)
- If optimal is identity, easier to learn F(x) = 0 than H(x) = x
- "It's easier to learn deviations from identity than to learn identity from scratch"

#### 3. **Ensemble-like Behavior**
- Network becomes ensemble of many shorter paths
- Each skip connection creates a shorter path through the network
- 2^n paths through network of depth n!

#### 4. **Better Initialization**
- At initialization, residual block ≈ identity
- Network starts with a "working" solution, then refines it

### Architecture Details
```
Input
  ↓
Conv 7×7, stride 2
  ↓
Max Pool
  ↓
[Residual Block] ×3   } Each block:
[Residual Block] ×4   } input → Conv → ReLU → Conv → (+input) → ReLU
[Residual Block] ×6   }
[Residual Block] ×3
  ↓
Global Average Pool
  ↓
FC + Softmax
```

### Why It Won ImageNet 2015
- **152 layers deep** (3.57% top-5 error)
- Previous winners: 16-30 layers
- Proved: With skip connections, deeper IS better

---

## Inception Module

### What is it?
**Run multiple convolution sizes in parallel**, then concatenate the outputs.

### Standard Inception Module Structure:
```
                    Input
                      |
        +-------------+-------------+
        |             |             |
     1×1 Conv      3×3 Conv      5×5 Conv     Max Pool
        |             |             |             |
        +-------------+-------------+-------------+
                      |
              Concatenate (depth-wise)
                      |
                   Output
```

### Why is it a Good Idea?

#### 1. **Multi-Scale Feature Extraction**
- Different kernel sizes capture different scales
- 1×1: Point-wise features
- 3×3: Local patterns
- 5×5: Larger context
- Let the network **learn which scales matter**

#### 2. **Increased Network Width**
- Traditional: Go deeper
- Inception: Go wider (multiple branches)
- More expressive without adding depth

#### 3. **Computational Efficiency** (with 1×1 convs)
- Problem: 5×5 convs are expensive
- Solution: 1×1 "bottleneck" layer before 3×3 and 5×5
```
Input (256 channels)
    ↓
  1×1 Conv (reduce to 64 channels)  ← Bottleneck
    ↓
  3×3 Conv (output 128 channels)
```
This dramatically reduces computation!

---

## Inception with Aggressive Factorization

### What is Aggressive Factorization?
Replace large convolutions with **sequences of smaller ones**.

### Example Factorizations:

#### 1. **5×5 → Two 3×3**
- One 5×5 conv ≈ Two 3×3 convs stacked
- Parameters: 25 → 18 (28% reduction)
- Receptive field: Same!
- More non-linearity (two ReLUs instead of one)

#### 2. **n×n → 1×n then n×1** (Asymmetric)
- 3×3 → 1×3 then 3×1
- Parameters: 9 → 6 (33% reduction)
- Example:
```
Input
  ↓
1×3 Conv
  ↓
3×1 Conv
  ↓
Output
```

### Why Do This?

#### 1. **Fewer Parameters**
- Less overfitting
- Faster training and inference

#### 2. **More Non-Linearity**
- More activation functions → more expressive
- Two ReLUs > one ReLU

#### 3. **Better Regularization**
- More layers = more regularization effect
- Similar to how ResNet benefits from depth

#### 4. **Computational Efficiency**
- Fewer FLOPs for same receptive field
- Can go deeper with same budget

### Where Used?
- **Inception-v2 and v3** heavily use factorization
- **Inception-v4** + ResNet hybrid
- Modern networks (EfficientNet) use similar ideas

---

## Squeeze Layer (Squeeze-and-Excitation)

### What is it?
A module that **recalibrates channel-wise feature responses** by explicitly modeling channel interdependencies.

### Architecture:
```
Input (H × W × C)
     ↓
Global Average Pool  → (1 × 1 × C)
     ↓
FC (C → C/r)        → Squeeze
     ↓
ReLU
     ↓
FC (C/r → C)        → Excitation
     ↓
Sigmoid             → (1 × 1 × C)
     ↓
Multiply with Input → (H × W × C)
```

### Step-by-Step:

#### 1. **Squeeze**: Global Information Embedding
- Global Average Pool across spatial dimensions
- Each channel → single number
- Captures global context

#### 2. **Excitation**: Channel-wise Attention
- Two FC layers (bottleneck with ratio r, typically 16)
- Sigmoid → outputs weights between 0 and 1
- One weight per channel

#### 3. **Recalibration**
- Multiply original features by channel weights
- Amplify useful channels, suppress less useful ones

### Why It Works:

#### **Channel Attention**
- Not all channels are equally important for a given input
- SE block learns to emphasize informative channels
- Adaptive recalibration

#### **Minimal Cost**
- Very few parameters (two small FC layers)
- Minimal computation (just global pooling + tiny FC)
- Big performance gain for small cost

### Where Used?
- **SENet** won ImageNet 2017
- Now added to many architectures (SE-ResNet, SE-Inception, etc.)
- MobileNetV3, EfficientNet use SE blocks

### Key Insight
**Attention mechanism for channels**: "Which channels should I pay attention to for this input?"
