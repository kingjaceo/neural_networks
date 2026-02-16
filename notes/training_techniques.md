# Training Techniques

## Initialization

### Why Good Initialization Matters
- Poor initialization → vanishing/exploding gradients
- Can determine if network trains at all
- Affects convergence speed

### Xavier/Glorot Initialization
- **For**: tanh, sigmoid activations
- **Formula**: Sample from uniform distribution with variance = 2/(n_in + n_out)
- **Goal**: Keep variance of activations and gradients similar across layers

### He Initialization
- **For**: ReLU and variants
- **Formula**: Sample from normal distribution with variance = 2/n_in
- **Why different**: ReLU kills half the neurons, so need more variance
- **Modern default**: Use this for CNNs with ReLU

### LeCun Initialization
- **For**: SELU activation
- **Formula**: Variance = 1/n_in
- **Less common** in modern practice

### Key Principle
Match the initialization to your activation function to maintain signal propagation through deep networks.

---

## Regularization

### What is Regularization?
Techniques to **prevent overfitting** by constraining the model's complexity.

### L2 Regularization (Weight Decay)
- **Add to loss**: λ · Σ(w²)
- **Effect**: Penalizes large weights → smoother functions
- **Most common**: λ ≈ 1e-4 to 1e-5

### L1 Regularization
- **Add to loss**: λ · Σ|w|
- **Effect**: Encourages sparsity (many weights → 0)
- **Less common** in deep learning

### Early Stopping
- Monitor validation loss, stop when it stops improving
- Simple and effective

### Data Augmentation
- Create variations of training data (flips, rotations, crops)
- **One of the most effective** regularization techniques for vision

---

## Dropout vs Drop Connection

### Dropout
- **What it removes**: **Neuron activations**
- **How**: During training, randomly set neurons to 0 with probability p
- **Effect**: Prevents co-adaptation of neurons
- **Training vs Test**:
  - Training: Multiply remaining activations by 1/(1-p) to maintain scale
  - Test: Use all neurons (no dropout)
- **Typical p**: 0.5 for FC layers, 0.1-0.2 for conv layers

### Drop Connection (DropConnect)
- **What it removes**: **Individual weights**
- **How**: During training, randomly set weights to 0 with probability p
- **Effect**: More fine-grained than dropout
- **Difference**: Dropout zeros entire neuron, DropConnect zeros individual connections

### Which is Better?
- **Dropout**: More popular, simpler, well-understood
- **DropConnect**: Theoretically more flexible, but more computation
- **Modern practice**: Dropout is standard; DropConnect used in specialized cases

---

## Batch Normalization

### What is it?
Normalize activations of each layer to have mean=0, std=1 **within each mini-batch**.

### Formula
```
x_norm = (x - μ_batch) / √(σ²_batch + ε)
y = γ · x_norm + β
```
Where γ and β are **learnable parameters**.

### Why Do We Use It?

#### 1. **Consistent Output Distribution Across Layers**
- Without it: As signals propagate, distributions shift wildly
- With it: Each layer gets normalized inputs → more stable training

#### 2. **Reduces Internal Covariate Shift**
- As earlier layers update, later layers' input distributions change
- BatchNorm stabilizes this

#### 3. **Practical Benefits**:
- **Faster training**: Can use higher learning rates
- **Regularization**: Acts as noise (each sample normalized by batch statistics)
- **Less sensitive** to initialization
- **Reduces need** for dropout

### Where to Place It?
```
Conv → BatchNorm → Activation (ReLU)
```
or
```
Conv → Activation (ReLU) → BatchNorm
```
Both work; first is more common.

### Training vs Inference
- **Training**: Use batch statistics (mean/std of current batch)
- **Inference**: Use running average of statistics from training

---

## Loss Functions

### Zero-One Loss Not Differentiable

#### What is Zero-One Loss?
```
L(y, ŷ) = { 0 if y = ŷ
          { 1 if y ≠ ŷ
```
Counts classification errors directly.

#### Why Not Differentiable?
- It's a **step function**
- Gradient is zero almost everywhere
- Can't use gradient descent!

#### What We Use Instead
**Surrogate losses** that are differentiable:
- **Cross-entropy** for classification (most common)
- **Hinge loss** for SVMs
- **Mean squared error** for regression

These upper-bound the zero-one loss and provide useful gradients.

---

## Monitoring Output as Regularization

### What Does This Mean?
Watching what your model outputs during training helps you regularize it properly.

### Practical Techniques:

#### 1. **Watch for Overconfidence**
- If softmax outputs are always 0.99+, model might be overfitting
- Solution: More regularization, label smoothing

#### 2. **Check Activation Statistics**
- Are all neurons firing? Or many dead (always 0)?
- Dead neurons → might need better initialization or different activation

#### 3. **Gradient Norms**
- Exploding gradients → gradient clipping needed
- Vanishing gradients → architecture or initialization issue

#### 4. **Validation Metrics**
- Gap between train and val accuracy → overfitting
- Both poor → underfitting or optimization issue

### Key Insight
Your model's outputs and internal states tell you **what's going wrong**, which tells you **what regularization to apply**.

---

## PCA for Removing Correlation

### What Was the Idea?
Use PCA to decorrelate features before feeding to neural network.

### Why We Don't Do This Anymore?
1. **Neural networks learn to decorrelate**: Early layers naturally learn useful transformations
2. **BatchNorm does this**: Normalizes and decorrelates to some extent
3. **End-to-end learning**: Let the network learn the right representation
4. **Loss of information**: PCA might throw away useful information

### When PCA Is Still Used:
- **Dimensionality reduction** for visualization
- **Pre-processing very high-dimensional data** (e.g., genomics)
- **Classical ML** (still common for SVMs, etc.)

But for deep learning: **Skip PCA, let the network learn.**
