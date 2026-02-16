# Convolutions and Pooling

## Same/Up/Down Convolution

### Same Convolution (Padding = 'same')
- **Goal**: Output has the **same spatial dimensions** as input
- **How**: Add padding around the input
- **Example**: For a 3×3 kernel, add 1 pixel of padding on all sides
- **Math**: If input is (H × W), output is (H × W)
- **Use case**: Preserve spatial resolution throughout the network

### Down Convolution (Stride > 1 or no padding)
- **Goal**: **Reduce** spatial dimensions
- **How**: Use stride > 1 or no padding
- **Example**: Stride 2 with 3×3 kernel → halves spatial dimensions
- **Math**: If input is (H × W) and stride=2, output is roughly (H/2 × W/2)
- **Use case**: Reduce computation, increase receptive field

### Up Convolution (Transposed Convolution / Deconvolution)
- **Goal**: **Increase** spatial dimensions
- **How**: Use transposed convolution (fractionally-strided convolution)
- **Example**: Transpose of a stride-2 conv → doubles spatial dimensions
- **Math**: If input is (H × W), output is roughly (2H × 2W)
- **Use case**: Upsampling in autoencoders, segmentation (U-Net), GANs

---

## Relationship Between Weights, Inputs, and Kernel Sizes

### The Connection
- **Kernel size** determines the **receptive field** (how much of the input each neuron "sees")
- **Number of input channels** determines depth of each kernel
- **Number of kernels** determines number of output channels

### Shape Mathematics
If you have:
- Input: (H × W × C_in)
- Kernel size: (K × K)
- Number of kernels: C_out

Then:
- **Each kernel shape**: (K × K × C_in)
- **All kernels together**: (K × K × C_in × C_out)
- **Output shape**: (H' × W' × C_out)

### Key Insight
The **C_in dimension** is where the convolution is actually summed:
```
output[i,j,k] = Σ Σ Σ (kernel[m,n,c,k] × input[i+m, j+n, c])
                m n c
```
This is why each kernel must have depth = C_in.

---

## Pooling Operations

### Max Pooling
- **Operation**: Take the **maximum** value in each pooling window
- **Window**: Typically 2×2 or 3×3
- **Stride**: Usually same as window size (non-overlapping)
- **Properties**:
  - Provides **translation invariance** (small shifts don't matter)
  - Keeps the **strongest activations**
  - Non-learnable (no parameters)
  - **Not differentiable** at non-max points (but we use subgradients)
- **Use case**: Most common in CNNs for downsampling

### Average Pooling
- **Operation**: Take the **average** value in each pooling window
- **Window**: Typically 2×2 or larger
- **Properties**:
  - Smooths the feature map
  - **Differentiable** everywhere
  - More gentle downsampling
  - Non-learnable (no parameters)
- **Use case**:
  - Sometimes used at the very end (Global Average Pooling)
  - GAP: Average over entire spatial dimensions → single value per channel
  - Reduces overfitting (fewer parameters than FC layers)

### Max vs Average
- **Max**: Preserves strong features, more aggressive
- **Average**: Preserves overall patterns, smoother
- **Modern trend**: Max pooling more common, or stride-2 convolutions to replace pooling entirely
