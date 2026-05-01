* Final Exam Topics: Lecture 13, Slide 13, through last lecture
* Self Attention Layer: review KEY, VALUE, QUERY
* position encoding with trig functions
* multi-head attention
* Variational Auto-Encoders and their Variants
   * Understand the relevance of distribution: are we training the representation space distribution? How can we sample from that?
   * why is batch normalization critical for capturing the correct distribution?
* GANs: how to sample from complex, high-dimensional training distribution (no direct way to do this) Solution: sample from a simple distribution (random noise), learn transformation to training distribution
* What is the adversarial part of GANs?
* GANs: why replace pooling layers with strided conv layers? why replace ReLU with Leaky ReLU (or any activation function w/ negative values)? why is batch normalization mandatory?
* Know how to compute KLD and JSD and why JSD is an improvement over KLD
* Lifelong Machine Learning: what is it?
* Knowledge distillation: how is it achieved? "focus only on the loss function"
* Backbone vs classifier
* Partial Network Sharing: how does it work
* iCaRL: what is it?
* GNN: graph neural network
* Why won't CNN and RNNs work on graph data?
* Equivariance, invariance, and permutaiton invariant
* Invariance: f(S(x)) = f(x), Equivariance: f(S(x)) = S(f(x)) - CNNs are invariant but not equivariant, GNNs are both (why??)  