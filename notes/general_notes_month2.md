# Neural Networks — Month 2 Notes

---

## Lecture 8 — Transfer Learning & Ensemble Methods

### Alternatives to Fully Connected Layers
- **Global Average Pooling (GAP)** — collapses spatial feature maps into a single vector; slashes parameter count and reduces overfitting
- **1×1 Convolution** — channel-wise projection; a leaner drop-in for the FC bottleneck

### Transfer Learning
- **Core idea** — instead of training from scratch, adapt a model pretrained on a large source task to your smaller target task
- **When to freeze vs. fine-tune** — small/similar dataset → freeze most layers; large/different dataset → fine-tune more (at a lower LR)
- **Negative transfer** — warning: domain mismatch can hurt more than it helps
- **Pretrained sources** — Model Zoo, HuggingFace (VGG, ResNet, Inception for vision; Word2Vec, BERT for NLP)

### Model Ensemble
- **Bagging** — train on bootstrapped subsets, average predictions; reduces variance
- **Boosting** — sequential training with re-weighted samples; reduces bias
- **Stacking** — meta-learner combines base model outputs; two-phase learning
- **Practical gain** — typically ~1–2% accuracy boost over a single model

---

## Lecture 9 — Segmentation / Regression Models

### Task Taxonomy
- **Semantic segmentation** — per-pixel class label (no instance distinction)
- **Instance segmentation** — per-pixel class *and* object identity
- **Panoramic segmentation** — union of the above; all objects, all instances

### Upsampling Primitives
- **Transposed convolution** — learnable upsampling; filter moves faster in output than input
- **Max unpooling** — remembers max locations from pooling, places values back there
- **Nearest-neighbor / bed-of-nails unpooling** — simpler, non-learnable alternatives

### Key Architectures
- **FCN (Fully Convolutional Network)** — first end-to-end pixel-prediction net; skip connections fuse coarse+fine features (FCN-32s → FCN-8s)
- **SegNet** — encoder-decoder with pooling-index upsampling; memory-efficient
- **U-Net** — symmetric encoder-decoder with skip connections; dominant in medical imaging
- **R2U-Net** — adds recurrent + residual units to U-Net; better feature representation at similar parameter cost
- **Dilated (Atrous) Convolution** — expands receptive field without downsampling; basis for DeepLab
- **DeepLab** — dilated convolutions + CRF post-processing for sharp boundaries
- **FastFCN** — replaces dilated backbone convolutions with Joint Pyramid Upsampling (JPU); same accuracy, faster
- **SAM (Segment Anything)** — foundation-model approach; prompt-driven universal segmentation

### Questions
- Why is segmentation a regression problem?

---

## Lecture 10 — Object Detection Models

### Problem Framing
- Detection = classification + regression: predict *what* the object is and *where* its bounding box is

### Two-Stage (Region Proposal) Detectors
- **R-CNN** — selective search proposals → per-region CNN → SVM + bbox regressor; accurate but slow (~2K forward passes)
- **Fast R-CNN** — single CNN over full image → RoI pooling; replaces SVM with softmax head
- **Faster R-CNN** — adds Region Proposal Network (RPN) to generate proposals inside the net; end-to-end, near real-time
- **R-FCN** — position-sensitive score maps; 100% shared computation across proposals

### Single-Stage Detectors
- **SSD** — simultaneous class + bbox prediction across multi-scale feature maps; uses hard negative mining and NMS
- **YOLO** — divides image into a grid; one forward pass predicts all boxes and classes; fastest

### Anchor-Free Detectors
- **FCOS** — fully convolutional; every point in a GT box is a positive sample; uses center-ness score to suppress low-quality boxes
- **FPN** - feature pyramid network
- **FoveaBox / CenterNet / CornerNet-Lite** — eliminate manual anchor hyperparameter tuning; mostly attention-based
- Why does multiplying center-ness by confidence produce explainability?

### Performance Metrics
- **IoU (Intersection over Union)** — primary metric for bounding box quality; threshold (e.g. 0.5) determines positive/negative

---

## Lecture 11 — Recurrent Neural Networks (RNN)

### Motivation
- **CNN/DNN limitation** — fixed-size inputs/outputs, purely feedforward, no temporal memory
- **RNN solution** — maintains a hidden state *h* that carries context across time steps; same weights reused at every step

### Core Mechanics
- **Recurrence formula** — `h_t = f_W(h_{t-1}, x_t)`; hidden state is a running summary of the sequence
- **Parameter sharing + unrolling** — one weight matrix W applied at every time step; enables arbitrary sequence lengths
- **BPTT (Backpropagation Through Time)** — treat the unrolled graph as a deep feedforward net; gradients flow back through all time steps

### Input–Output Configurations
- **One-to-one** — standard classification (degenerate case)
- **One-to-many** — image captioning
- **Many-to-one** — sentiment classification, sequence labeling
- **Many-to-many** — machine translation (seq2seq: encoder compresses → decoder generates)

### RNN Architectures
- **Elman RNN** — hidden-to-hidden connections (most common "vanilla" RNN)
- **Jordan RNN** — output fed back instead of hidden state
- **Hopfield Network** — fully connected, energy-based associative memory

### Applications
- Language modeling, machine translation, speech recognition, music generation
- Image captioning (CNN encoder + RNN decoder)
- Sentiment analysis, time-series forecasting, bioinformatics (DNA/protein sequences)

### What's Next
- **LSTMs & GRUs** — gated variants that solve the vanishing gradient problem in long sequences
