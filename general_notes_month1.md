**Topics/Notes to Review**
- Same/up/down convolution
- Squeeze Layer
- How do you initialize a NN well?
- Activation Functions: switch/mesh/GELU?
- Regularization
- "Drop out" vs "Drop connection": removing (?) neurons vs weights
- Remove correlation w/ PCA (we don't do this anymore?)
- "Batch normalization"? For feature representation? "Make output distribution consistent across layers". Why?
- Zero-one loss not differentiable
- Average pooling
- ResNet: why is connecting the inputs to the output ReLU a good idea?
- Max pooling
- S3/S4 activation function
- Inception Module: what is it and why is it a good idea?
- "Monitoring output is part of regularizing your model"
- Inception with "agressive factorization of features": why?
- Relationship between weights, inputs, and convolutions (or kernel sizes?)
- How do you choose 64 kernels? Does anyone choose 1024 kernels? How?

**2/11 Notes**
- Res2Net
- Averaging output layer and resizing it to create a heatmap for "explainability" (ex: (8x8x1024) output -> (8x8) average -> (224x224x3) input)
- PolyNet
- FractalNet
- DenseNet
- Vanishing Gradient Problem
- Feature Propagation
- Are kernels hard-coded? Or learned?
- What would a 1x1 kernel be or do?