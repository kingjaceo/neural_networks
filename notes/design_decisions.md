# Design Decisions and Hyperparameters

## How Do You Choose Number of Kernels?

### The Question
"How do you choose 64 kernels? Does anyone choose 1024 kernels? How?"

### Common Patterns in CNN Architectures

#### **Early Layers**: Fewer kernels
- First conv layer: 32-64 kernels
- Reason: Input has few channels (3 for RGB), detecting simple features (edges, colors)

#### **Middle Layers**: Progressively increase
- Common progression: 64 → 128 → 256 → 512
- Often **double** after each pooling/downsampling
- Reason: Spatial dimensions shrink, can afford more channels

#### **Deep Layers**: Many kernels
- Yes, 512-1024+ kernels is common in deep networks!
- VGG: goes up to 512
- ResNet-50: goes up to 2048
- EfficientNet-B7: goes up to 2560

### Why This Pattern Works

#### **Computation Trade-off**
Early in network:
- Spatial dimensions: Large (e.g., 224×224)
- Channels: Small (e.g., 64)
- Computation: (224×224×64) = large spatial

Later in network:
- Spatial dimensions: Small (e.g., 7×7)
- Channels: Large (e.g., 512)
- Computation: (7×7×512) ≈ similar cost!

#### **Feature Hierarchy**
- Early: Few types of simple features (edges, colors)
- Late: Many types of complex features (object parts, textures, patterns)
- Need more kernels to capture diverse high-level concepts

### How Do You Actually Choose?

#### 1. **Follow Established Architectures**
- Start with proven patterns (ResNet, EfficientNet)
- For custom tasks, use similar scaling

#### 2. **Double After Downsampling**
- Standard rule: When spatial dims halve, double channels
- Keeps computational cost roughly constant
- Example: (56×56×128) → pool → (28×28×256)

#### 3. **Consider Your Dataset**
- Small dataset (CIFAR-10): 32→64→128 might be enough
- Large dataset (ImageNet): 64→128→256→512→1024
- More data → can support more parameters

#### 4. **Computational Budget**
- Mobile/edge: Smaller (e.g., MobileNet uses depthwise separable convs)
- Server/GPU: Larger (e.g., EfficientNet scales up)

#### 5. **Empirical Tuning**
- Grid search or neural architecture search (NAS)
- But usually, following patterns works well

### Typical Scaling Examples

#### **VGG-16**:
```
64 → 128 → 256 → 512 → 512
```

#### **ResNet-50**:
```
64 → 64 → 128 → 256 → 512 → 2048 (at final layers)
```

#### **EfficientNet-B0**:
```
32 → 16 → 24 → 40 → 80 → 112 → 192 → 320 → 1280
```
(Uses compound scaling: depth, width, resolution together)

### Modern Approaches

#### **Neural Architecture Search (NAS)**
- Automatically find optimal channel numbers
- Used for EfficientNet, MobileNetV3
- Expensive but effective

#### **Compound Scaling**
- Scale depth, width (channels), and resolution together
- EfficientNet: Use a coefficient α, β, γ
  - Depth: α^φ
  - Width: β^φ
  - Resolution: γ^φ

#### **Width Multiplier**
- MobileNet: Use α to scale all channels
- α=0.5 → half the channels → quarter the cost
- Flexible for different resource constraints

---

## General Design Principles

### Start with a Baseline
1. Use established architecture as starting point
2. Understand why it works
3. Modify for your specific task

### Iterate Based on Performance
- **Underfitting** (train accuracy low):
  - Increase capacity (more channels, deeper)
  - Train longer
- **Overfitting** (train good, val poor):
  - Regularization (dropout, weight decay)
  - Data augmentation
  - Reduce capacity

### Consider the Full Pipeline
- Channels are just one dimension
- Also consider: depth, kernel sizes, skip connections, batch size, learning rate
- **Everything interacts**: More channels → might need lower learning rate

---

## Key Takeaway

**There's no magic formula**, but there are strong patterns:
- Follow established architectures for your domain
- Early layers: fewer channels, larger spatial
- Late layers: more channels, smaller spatial
- Double channels when you halve spatial dimensions
- Tune based on dataset size and computational budget

**Yes, people use 1024+ kernels!** It's standard in deep networks for large-scale tasks.
