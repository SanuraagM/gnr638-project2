"""
Generates 100 NEW Deep Learning MCQ images (synthetic_test3).
All questions are different from synthetic_test and synthetic_test2.
"""
import os
import csv
import subprocess
import shutil
import tempfile
import random
from pdf2image import convert_from_path

MCQS = [
    # ── THEORETICAL (1-40) ──────────────────────────────────────────────
    {"type": "theoretical", "title": "Layer Normalization",
     "body": "Layer Normalization differs from Batch Normalization in that it normalizes:",
     "options": ["Across the batch dimension",
                 "Across all features for each individual sample",
                 "Only the convolutional layers",
                 "Only during inference"], "answer": 2},

    {"type": "theoretical", "title": "BERT Pretraining",
     "body": "BERT is pretrained using which two tasks?",
     "options": ["Next sentence prediction and image captioning",
                 "Masked language modeling and next sentence prediction",
                 "Text classification and token prediction",
                 "Question answering and summarization"], "answer": 2},

    {"type": "theoretical", "title": "Dilated Convolution",
     "body": "Dilated (atrous) convolutions increase the receptive field without:",
     "options": ["Using any parameters",
                 "Increasing the number of layers",
                 "Increasing parameters or losing spatial resolution",
                 "Using padding"], "answer": 3},

    {"type": "theoretical", "title": "Label Smoothing",
     "body": "Label smoothing regularization works by:",
     "options": ["Adding noise to input images",
                 "Replacing hard 0/1 targets with soft values like 0.1/0.9",
                 "Randomly dropping labels during training",
                 "Increasing the number of output classes"], "answer": 2},

    {"type": "theoretical", "title": "Group Normalization",
     "body": "Group Normalization divides channels into groups and normalizes within each group. It is preferred over Batch Normalization when:",
     "options": ["Batch size is very large",
                 "Batch size is very small (e.g., 1 or 2)",
                 "Training on TPUs",
                 "Using sigmoid activations"], "answer": 2},

    {"type": "theoretical", "title": "Mixup Augmentation",
     "body": "Mixup data augmentation creates training samples by:",
     "options": ["Randomly cropping images",
                 "Linearly interpolating between two training examples and their labels",
                 "Adding Gaussian noise to images",
                 "Flipping images horizontally"], "answer": 2},

    {"type": "theoretical", "title": "Cosine Annealing",
     "body": "Cosine annealing as a learning rate schedule:",
     "options": ["Keeps the learning rate constant throughout training",
                 "Decays the learning rate following a cosine curve",
                 "Increases the learning rate at each step",
                 "Randomly sets the learning rate"], "answer": 2},

    {"type": "theoretical", "title": "Gradient Checkpointing",
     "body": "Gradient checkpointing reduces GPU memory usage during training by:",
     "options": ["Using smaller batch sizes automatically",
                 "Recomputing intermediate activations during the backward pass instead of storing them",
                 "Pruning unused gradients",
                 "Quantizing gradients to 8-bit"], "answer": 2},

    {"type": "theoretical", "title": "Curriculum Learning",
     "body": "In curriculum learning, training data is presented:",
     "options": ["Randomly throughout training",
                 "From hardest to easiest examples",
                 "From easiest to hardest examples",
                 "Only once per epoch in fixed order"], "answer": 3},

    {"type": "theoretical", "title": "Spectral Normalization",
     "body": "Spectral normalization in GANs constrains the:",
     "options": ["Gradient norm of the generator",
                 "Lipschitz constant of the discriminator by normalizing weight matrices",
                 "Output range of the generator",
                 "Learning rate of the discriminator"], "answer": 2},

    {"type": "theoretical", "title": "Few-Shot Learning",
     "body": "In a 5-way 1-shot learning task, the model must:",
     "options": ["Train on 5 examples per class",
                 "Classify into 5 classes using only 1 labeled example per class",
                 "Use 5 different models",
                 "Complete 1 task with 5 augmented samples"], "answer": 2},

    {"type": "theoretical", "title": "Anchor Boxes",
     "body": "Anchor boxes in object detection frameworks like YOLO are used to:",
     "options": ["Normalize image inputs",
                 "Predict objects of different scales and aspect ratios",
                 "Apply non-maximum suppression",
                 "Extract image features"], "answer": 2},

    {"type": "theoretical", "title": "ROI Pooling",
     "body": "ROI Pooling in Faster R-CNN converts region proposals of different sizes into:",
     "options": ["Variable size feature maps",
                 "Fixed size feature maps for the classification head",
                 "Binary masks for segmentation",
                 "Bounding box offsets"], "answer": 2},

    {"type": "theoretical", "title": "Mixed Precision Training",
     "body": "Mixed precision training uses FP16 for most computations to:",
     "options": ["Improve model accuracy",
                 "Reduce memory usage and speed up training while maintaining FP32 for critical ops",
                 "Enable training without a GPU",
                 "Replace batch normalization"], "answer": 2},

    {"type": "theoretical", "title": "CutMix Augmentation",
     "body": "CutMix augmentation differs from Mixup in that it:",
     "options": ["Blends two entire images pixel-by-pixel",
                 "Cuts and pastes patches between training images, mixing labels proportionally",
                 "Randomly erases rectangular regions",
                 "Applies color jitter to images"], "answer": 2},

    {"type": "theoretical", "title": "Transformer Encoder vs Decoder",
     "body": "The Transformer decoder differs from the encoder by having an additional:",
     "options": ["Self-attention layer",
                 "Cross-attention layer attending to encoder outputs",
                 "Feed-forward layer",
                 "Positional encoding layer"], "answer": 2},

    {"type": "theoretical", "title": "Neural Architecture Search",
     "body": "Neural Architecture Search (NAS) aims to:",
     "options": ["Automatically find optimal hyperparameters like learning rate",
                 "Automatically design neural network architectures",
                 "Search for the best training dataset",
                 "Find optimal data augmentation strategies"], "answer": 2},

    {"type": "theoretical", "title": "Quantization",
     "body": "Post-training quantization reduces model size by:",
     "options": ["Removing entire layers from the network",
                 "Representing weights and activations with lower bit precision (e.g., INT8)",
                 "Pruning neurons with small weights",
                 "Distilling the model into a smaller one"], "answer": 2},

    {"type": "theoretical", "title": "Deformable Convolution",
     "body": "Deformable convolutions improve upon standard convolutions by:",
     "options": ["Using larger kernel sizes",
                 "Learning offsets that allow the kernel to adapt its sampling locations",
                 "Processing multiple scales simultaneously",
                 "Reducing computation by sharing weights"], "answer": 2},

    {"type": "theoretical", "title": "Graph Neural Networks",
     "body": "Graph Neural Networks (GNNs) operate on graph-structured data by:",
     "options": ["Flattening the graph into a sequence",
                 "Aggregating information from neighboring nodes",
                 "Converting graphs to images first",
                 "Using recurrent connections"], "answer": 2},

    {"type": "theoretical", "title": "Warmup Scheduling",
     "body": "Learning rate warmup at the start of training is used to:",
     "options": ["Achieve faster convergence from the beginning",
                 "Prevent instability caused by large gradients early in training",
                 "Increase the model's batch size gradually",
                 "Pre-train specific layers"], "answer": 2},

    {"type": "theoretical", "title": "Capsule Networks",
     "body": "Capsule Networks were proposed to address a key limitation of CNNs, which is:",
     "options": ["Slow inference speed",
                 "Poor viewpoint invariance and lack of pose information",
                 "High memory consumption",
                 "Inability to use batch normalization"], "answer": 2},

    {"type": "theoretical", "title": "Squeeze-and-Excitation",
     "body": "Squeeze-and-Excitation (SE) blocks improve CNNs by:",
     "options": ["Adding residual connections",
                 "Adaptively recalibrating channel-wise feature responses",
                 "Applying spatial attention to feature maps",
                 "Replacing pooling with strided convolutions"], "answer": 2},

    {"type": "theoretical", "title": "Attention vs Convolution",
     "body": "Compared to convolutions, self-attention can capture:",
     "options": ["Only local spatial features",
                 "Long-range dependencies across the entire sequence",
                 "Only channel-wise features",
                 "Hierarchical spatial features only"], "answer": 2},

    {"type": "theoretical", "title": "Zero-Shot Learning",
     "body": "Zero-shot learning enables a model to classify:",
     "options": ["Classes with zero training examples using semantic descriptions",
                 "All classes using zero GPU memory",
                 "Images with zero preprocessing",
                 "Tasks with zero labeled data using reinforcement"], "answer": 1},

    {"type": "theoretical", "title": "Pruning",
     "body": "Unstructured pruning in neural networks removes:",
     "options": ["Entire layers from the network",
                 "Individual weights below a certain magnitude threshold",
                 "Entire filters from convolutional layers",
                 "Neurons from fully connected layers only"], "answer": 2},

    {"type": "theoretical", "title": "Mean Average Precision",
     "body": "In object detection, mAP (mean Average Precision) is computed as:",
     "options": ["Average accuracy across all test images",
                 "Mean of AP values computed per class across IoU thresholds",
                 "Maximum precision at 50\\% recall",
                 "Average loss across all detection heads"], "answer": 2},

    {"type": "theoretical", "title": "Panoptic Segmentation",
     "body": "Panoptic segmentation combines:",
     "options": ["Object detection and depth estimation",
                 "Semantic segmentation and instance segmentation",
                 "Image classification and pose estimation",
                 "Object tracking and segmentation"], "answer": 2},

    {"type": "theoretical", "title": "Siamese Networks",
     "body": "Siamese networks are used primarily for:",
     "options": ["Multi-class image classification",
                 "Similarity learning between pairs of inputs",
                 "Generative image synthesis",
                 "Object detection in video"], "answer": 2},

    {"type": "theoretical", "title": "ViT Patch Embedding",
     "body": "Vision Transformer (ViT) splits an image into fixed-size patches and treats them as:",
     "options": ["Convolutional feature maps",
                 "Tokens (similar to words in NLP)",
                 "Anchor boxes for detection",
                 "Probability distributions"], "answer": 2},

    {"type": "theoretical", "title": "Contrastive Loss vs Triplet Loss",
     "body": "Triplet loss trains with (anchor, positive, negative) to ensure:",
     "options": ["Anchor is closer to negative than positive",
                 "Anchor is closer to positive than negative by a margin",
                 "All three samples are equidistant",
                 "Negative is the hardest example in the batch"], "answer": 2},

    {"type": "theoretical", "title": "Conditional GAN",
     "body": "In a Conditional GAN (cGAN), the generator receives:",
     "options": ["Only a noise vector",
                 "A noise vector and a class label or condition",
                 "Only the class label",
                 "Real images and noise"], "answer": 2},

    {"type": "theoretical", "title": "Batch Size Effect",
     "body": "Training with very large batch sizes tends to:",
     "options": ["Always improve generalization",
                 "Converge to sharper minima with potentially worse generalization",
                 "Reduce training time with no other effects",
                 "Always reduce the final loss"], "answer": 2},

    {"type": "theoretical", "title": "Transposed Convolution",
     "body": "Transposed convolution (deconvolution) is commonly used in:",
     "options": ["Compressing feature maps",
                 "Upsampling feature maps in decoder architectures",
                 "Replacing pooling layers in encoders",
                 "Computing attention scores"], "answer": 2},

    {"type": "theoretical", "title": "t-SNE Purpose",
     "body": "t-SNE is primarily used for:",
     "options": ["Training better embeddings",
                 "Visualizing high-dimensional data in 2D or 3D",
                 "Regularizing neural networks",
                 "Augmenting training data"], "answer": 2},

    {"type": "theoretical", "title": "Dropout at Inference",
     "body": "During inference, dropout is typically:",
     "options": ["Applied with the same probability as training",
                 "Disabled, and weights are scaled by $(1-p)$",
                 "Applied at half the training probability",
                 "Applied only to the first layer"], "answer": 2},

    {"type": "theoretical", "title": "InstanceNorm in Style Transfer",
     "body": "Instance Normalization is preferred over Batch Normalization in style transfer because:",
     "options": ["It normalizes across the batch for consistent style",
                 "It normalizes per-image, preserving per-instance style statistics",
                 "It uses learnable parameters that encode style",
                 "It is faster to compute"], "answer": 2},

    {"type": "theoretical", "title": "Gradient Flow in ResNet",
     "body": "In ResNets, the skip connection ensures gradients can flow:",
     "options": ["Only through the residual branch",
                 "Directly from later layers to earlier layers via the identity path",
                 "Only through the final layer",
                 "Backwards through batch normalization only"], "answer": 2},

    {"type": "theoretical", "title": "CLIP Training",
     "body": "OpenAI's CLIP model is trained using:",
     "options": ["Supervised labels on ImageNet",
                 "Contrastive loss between paired image and text embeddings",
                 "Masked image modeling",
                 "Generative adversarial training"], "answer": 2},

    {"type": "theoretical", "title": "Attention Mask",
     "body": "A causal (or autoregressive) attention mask in GPT ensures:",
     "options": ["Each token attends to all other tokens",
                 "Each token can only attend to previous tokens, not future ones",
                 "Padding tokens are ignored",
                 "Attention weights sum to zero"], "answer": 2},

    # ── EQUATION-BASED (41-70) ──────────────────────────────────────────
    {"type": "equation", "title": "Receptive Field Calculation",
     "body": r"""Two consecutive Conv2D layers each with kernel size $3\times3$ and stride $1$, no padding.
What is the effective receptive field on the input after both layers?""",
     "options": ["$3 \\times 3$", "$5 \\times 5$", "$6 \\times 6$", "$9 \\times 9$"], "answer": 2},

    {"type": "equation", "title": "Hinge Loss",
     "body": r"""The hinge loss for SVM is: $\mathcal{L} = \max(0,\; 1 - y \cdot \hat{y})$

For $y = 1$ and $\hat{y} = 0.6$, what is the loss?""",
     "options": ["$0.6$", "$0.4$", "$0$", "$1.6$"], "answer": 2},

    {"type": "equation", "title": "Cosine Similarity",
     "body": r"""Cosine similarity: $\cos(\theta) = \dfrac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$

For $\mathbf{a}=[1,0]$ and $\mathbf{b}=[0,1]$, what is $\cos(\theta)$?""",
     "options": ["$1$", "$-1$", "$0$", "$0.5$"], "answer": 3},

    {"type": "equation", "title": "Adam First Moment",
     "body": r"""Adam first moment update: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$

With $\beta_1=0.9$, $m_0=0$, $g_1=1$. What is $m_1$?""",
     "options": ["$0.9$", "$0.1$", "$1.0$", "$0.01$"], "answer": 2},

    {"type": "equation", "title": "Conv FLOPs",
     "body": r"""FLOPs for one Conv2D layer:
$$\text{FLOPs} \approx 2 \times C_{\text{in}} \times C_{\text{out}} \times K^2 \times H_{\text{out}} \times W_{\text{out}}$$
For $C_{\text{in}}=3$, $C_{\text{out}}=8$, $K=3$, $H_{\text{out}}=W_{\text{out}}=16$, what are the FLOPs?""",
     "options": ["$27{,}648$", "$55{,}296$", "$13{,}824$", "$110{,}592$"], "answer": 2},

    {"type": "equation", "title": "Top-k Accuracy",
     "body": r"""Top-$k$ accuracy counts a prediction as correct if the true label is among the top-$k$ predicted classes.

For $k=1$ and softmax outputs $[0.1, 0.6, 0.2, 0.1]$ with true label class 2, is the prediction correct?""",
     "options": ["Yes, class 2 is the top-1 prediction",
                 "No, class 2 is not the top-1 prediction",
                 "Yes, but only for $k=2$",
                 "Cannot determine without logits"], "answer": 1},

    {"type": "equation", "title": "Dice Loss",
     "body": r"""Dice loss: $\mathcal{L}_{\text{dice}} = 1 - \dfrac{2|A \cap B|}{|A|+|B|}$

For $|A \cap B|=8$, $|A|=10$, $|B|=10$, what is the Dice loss?""",
     "options": ["$0.8$", "$0.2$", "$0.1$", "$0.6$"], "answer": 2},

    {"type": "equation", "title": "IoU Calculation",
     "body": r"""Intersection over Union (IoU):
$$\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$
If intersection area $= 4$ and union area $= 16$, what is IoU?""",
     "options": ["$0.5$", "$0.25$", "$0.75$", "$4$"], "answer": 2},

    {"type": "equation", "title": "Huber Loss",
     "body": r"""Huber loss with $\delta=1$:
$$\mathcal{L}_\delta(a) = \begin{cases} \frac{1}{2}a^2 & |a| \le 1 \\ |a| - \frac{1}{2} & |a| > 1 \end{cases}$$
For $a = 2$, what is $\mathcal{L}_\delta(2)$?""",
     "options": ["$2.0$", "$1.5$", "$4.0$", "$0.5$"], "answer": 2},

    {"type": "equation", "title": "Dropout Test Time Scaling",
     "body": r"""During training with dropout rate $p=0.4$, outputs are scaled by $\dfrac{1}{1-p}$.

If a neuron has output $x=5$ during training (after scaling), what is the raw expected value?""",
     "options": ["$5.0$", "$3.0$", "$2.0$", "$8.33$"], "answer": 2},

    {"type": "equation", "title": "Batch Norm Scale Parameter",
     "body": r"""After batch normalization, the output is: $y = \gamma \hat{x} + \beta$

If $\gamma=2$, $\beta=1$, and $\hat{x}=0$, what is $y$?""",
     "options": ["$0$", "$1$", "$2$", "$3$"], "answer": 2},

    {"type": "equation", "title": "Perceptron Update",
     "body": r"""The perceptron update rule is: $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y \cdot \mathbf{x}$

With $\mathbf{w}=[0,0]$, $\eta=1$, $y=1$, $\mathbf{x}=[2,3]$. What is the new $\mathbf{w}$?""",
     "options": ["$[2, 3]$", "$[1, 1]$", "$[0, 0]$", "$[-2, -3]$"], "answer": 1},

    {"type": "equation", "title": "Triplet Loss",
     "body": r"""Triplet loss: $\mathcal{L} = \max\!\bigl(0,\; d(a,p) - d(a,n) + \alpha\bigr)$

With $d(a,p)=0.3$, $d(a,n)=0.8$, $\alpha=0.2$. What is the loss?""",
     "options": ["$0$", "$0.2$", "$0.7$", "$0.3$"], "answer": 1},

    {"type": "equation", "title": "Precision and Recall",
     "body": r"""For a binary classifier: $TP=90$, $FP=10$, $FN=5$, $TN=95$.

What is the Precision?""",
     "options": ["$0.90$", "$0.95$", "$0.85$", "$0.94$"], "answer": 1},

    {"type": "equation", "title": "Convolution Padding for Same Size",
     "body": r"""Given input size $W=100$, kernel $K=7$, stride $S=1$.
What padding $P$ gives output size equal to input size?""",
     "options": ["$P=2$", "$P=3$", "$P=4$", "$P=6$"], "answer": 2},

    {"type": "equation", "title": "Gradient of Cross-Entropy + Softmax",
     "body": r"""For cross-entropy loss with softmax, the gradient with respect to logit $z_i$ is:
$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$$
If $p_i=0.7$ and $y_i=1$, what is the gradient?""",
     "options": ["$0.7$", "$-0.3$", "$0.3$", "$1.7$"], "answer": 2},

    {"type": "equation", "title": "Linear Layer Output Shape",
     "body": r"""A batch of shape $(32, 512)$ is passed through:
\texttt{nn.Linear(512, 256)}.
What is the output shape?""",
     "options": ["$(32, 512)$", "$(32, 256)$", "$(256, 512)$", "$(512, 256)$"], "answer": 2},

    {"type": "equation", "title": "Encoder Output Sequence Length",
     "body": r"""A Transformer encoder takes a sequence of $L=10$ tokens, each embedded to $d=512$.
What is the shape of the encoder output?""",
     "options": ["$(10, 512)$", "$(512, 10)$", "$(10, 10)$", "$(1, 512)$"], "answer": 1},

    {"type": "equation", "title": "Attention Output Shape",
     "body": r"""Scaled dot-product attention takes $Q, K, V$ of shape $(n, d_k)$, $(n, d_k)$, $(n, d_v)$.
What is the shape of the attention output?""",
     "options": ["$(n, d_k)$", "$(n, d_v)$", "$(d_k, d_v)$", "$(n, n)$"], "answer": 2},

    {"type": "equation", "title": "Total Parameters in Linear Stack",
     "body": r"""Two linear layers with bias:
\begin{itemize}
  \item Linear(128, 64)
  \item Linear(64, 10)
\end{itemize}
Total trainable parameters?""",
     "options": ["$8{,}896$", "$9{,}034$", "$8{,}320$", "$9{,}290$"], "answer": 1},

    {"type": "equation", "title": "Strided Conv Output",
     "body": r"""Input: $56 \times 56$, Conv2D: kernel $1\times1$, stride $2$, padding $0$.
Output spatial size?""",
     "options": ["$56 \times 56$", "$28 \times 28$", "$27 \times 27$", "$14 \times 14$"], "answer": 2},

    {"type": "equation", "title": "Softmax Temperature Effect",
     "body": r"""Given logits $z = [4, 1, 1]$. As temperature $T \to \infty$, softmax output approaches:""",
     "options": ["$[1, 0, 0]$", "$[0.33, 0.33, 0.33]$", "$[0, 0, 1]$", "$[4, 1, 1]$"], "answer": 2},

    {"type": "equation", "title": "Depthwise Separable Parameters",
     "body": r"""Standard Conv2D: $C_{in}=32$, $C_{out}=64$, $K=3$. Depthwise separable splits into depthwise + pointwise.
How many parameters total (no bias)?""",
     "options": ["$18{,}432$", "$2{,}336$", "$2{,}048$", "$4{,}608$"], "answer": 2},

    {"type": "equation", "title": "KL Divergence Asymmetry",
     "body": r"""KL divergence: $D_{KL}(P\|Q) = \sum_x P(x)\log\dfrac{P(x)}{Q(x)}$

Which statement is correct?""",
     "options": ["$D_{KL}(P\|Q) = D_{KL}(Q\|P)$ always",
                 "$D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ in general",
                 "$D_{KL}(P\|Q) < 0$ when $P \neq Q$",
                 "$D_{KL}(P\|Q) = 1$ when $P = Q$"], "answer": 2},

    {"type": "equation", "title": "Recall Calculation",
     "body": r"""Recall $= \dfrac{TP}{TP + FN}$

For $TP=70$, $FP=30$, $FN=10$, $TN=90$, what is Recall?""",
     "options": ["$0.70$", "$0.875$", "$0.78$", "$0.90$"], "answer": 2},

    {"type": "equation", "title": "AdaGrad Update",
     "body": r"""AdaGrad update: $\theta \leftarrow \theta - \dfrac{\eta}{\sqrt{G_t + \epsilon}} g_t$

If $G_t=3$, $\epsilon=1$, $\eta=2$, $g_t=1$, what is the effective step size?""",
     "options": ["$2.0$", "$1.0$", "$0.5$", "$4.0$"], "answer": 2},

    {"type": "equation", "title": "Conv2D with Dilation",
     "body": r"""For a dilated Conv2D with kernel $K=3$, dilation $d=2$, the effective kernel size is:
$$K_{\text{eff}} = K + (K-1)(d-1)$$
What is $K_{\text{eff}}$?""",
     "options": ["$3$", "$5$", "$7$", "$9$"], "answer": 2},

    {"type": "equation", "title": "Feature Map Memory",
     "body": r"""A feature map of shape $(64, 256, 56, 56)$ stored in float32.
How much memory does it use in MB?""",
     "options": ["$128$ MB", "$512$ MB", "$256$ MB", "$1024$ MB"], "answer": 2},

    {"type": "equation", "title": "Average Pooling Output",
     "body": r"""AvgPool2D with kernel $3\times3$, stride $1$, padding $0$ applied to $9\times9$ feature map.
Output spatial size?""",
     "options": ["$9 \times 9$", "$7 \times 7$", "$3 \times 3$", "$6 \times 6$"], "answer": 2},

    {"type": "equation", "title": "Weight Init Variance",
     "body": r"""He (Kaiming) initialization sets weight variance to:
$$\text{Var}(w) = \frac{2}{n_{\text{in}}}$$
For $n_{\text{in}}=512$, what is the standard deviation?""",
     "options": ["$0.0625$", "$0.0625$", "$\\approx 0.0625$", "$\\approx 0.063$"], "answer": 4},

    # ── CODE SNIPPET (71-100) ───────────────────────────────────────────
    {"type": "code", "title": "LSTM Output Shape",
     "body": r"What is the shape of \texttt{out}?",
     "code_options": [
         "lstm = nn.LSTM(10, 20, batch_first=True)\nx = torch.randn(4, 5, 10)\nout, (h, c) = lstm(x)\n# out shape = (4, 20)",
         "lstm = nn.LSTM(10, 20, batch_first=True)\nx = torch.randn(4, 5, 10)\nout, (h, c) = lstm(x)\n# out shape = (4, 5, 20)",
         "lstm = nn.LSTM(10, 20, batch_first=True)\nx = torch.randn(4, 5, 10)\nout, (h, c) = lstm(x)\n# out shape = (5, 4, 10)",
         "lstm = nn.LSTM(10, 20, batch_first=True)\nx = torch.randn(4, 5, 10)\nout, (h, c) = lstm(x)\n# out shape = (4, 10, 20)",
     ], "answer": 2},

    {"type": "code", "title": "Custom Loss Function",
     "body": r"Which custom loss correctly implements MSE?",
     "code_options": [
         "def my_loss(pred, target):\n    return (pred - target).sum()",
         "def my_loss(pred, target):\n    return ((pred - target) ** 2).mean()",
         "def my_loss(pred, target):\n    return (pred - target).abs().mean()",
         "def my_loss(pred, target):\n    return (pred * target).mean()",
     ], "answer": 2},

    {"type": "code", "title": "Gradient Clipping",
     "body": r"Which line clips gradients to max norm $1.0$?",
     "code_options": [
         "loss.backward()\ntorch.nn.utils.clip_grad_value_(model.parameters(), 1.0)\noptimizer.step()",
         "loss.backward()\ntorch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\noptimizer.step()",
         "loss.backward()\nmodel.clip_gradients(1.0)\noptimizer.step()",
         "loss.backward()\noptimizer.clip(1.0)\noptimizer.step()",
     ], "answer": 2},

    {"type": "code", "title": "nn.ModuleList",
     "body": r"Why use \texttt{nn.ModuleList} instead of a Python list?",
     "code_options": [
         "layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(5)])\n# A: It is faster than Python list",
         "layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(5)])\n# B: Parameters are registered and appear in model.parameters()",
         "layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(5)])\n# C: It allows dynamic graph construction",
         "layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(5)])\n# D: It shares weights between layers",
     ], "answer": 2},

    {"type": "code", "title": "torch.einsum Matmul",
     "body": r"What does this \texttt{einsum} compute?",
     "code_options": [
         "A = torch.randn(3, 4)\nB = torch.randn(4, 5)\nC = torch.einsum('ij,jk->ik', A, B)\n# A: Element-wise product",
         "A = torch.randn(3, 4)\nB = torch.randn(4, 5)\nC = torch.einsum('ij,jk->ik', A, B)\n# B: Matrix multiplication, shape (3,5)",
         "A = torch.randn(3, 4)\nB = torch.randn(4, 5)\nC = torch.einsum('ij,jk->ik', A, B)\n# C: Transpose of A times B",
         "A = torch.randn(3, 4)\nB = torch.randn(4, 5)\nC = torch.einsum('ij,jk->ik', A, B)\n# D: Outer product, shape (3,4,5)",
     ], "answer": 2},

    {"type": "code", "title": "F.interpolate Upsample",
     "body": r"What is the shape of \texttt{y}?",
     "code_options": [
         "x = torch.randn(2, 16, 8, 8)\ny = F.interpolate(x, scale_factor=2, mode='nearest')\n# shape = (2, 16, 8, 8)",
         "x = torch.randn(2, 16, 8, 8)\ny = F.interpolate(x, scale_factor=2, mode='nearest')\n# shape = (2, 16, 16, 16)",
         "x = torch.randn(2, 16, 8, 8)\ny = F.interpolate(x, scale_factor=2, mode='nearest')\n# shape = (2, 32, 8, 8)",
         "x = torch.randn(2, 16, 8, 8)\ny = F.interpolate(x, scale_factor=2, mode='nearest')\n# shape = (4, 16, 8, 8)",
     ], "answer": 2},

    {"type": "code", "title": "nn.LayerNorm",
     "body": r"What does \texttt{nn.LayerNorm(512)} normalize over?",
     "code_options": [
         "x = torch.randn(8, 10, 512)\nnorm = nn.LayerNorm(512)\ny = norm(x)\n# A: Across batch dimension (dim=0)",
         "x = torch.randn(8, 10, 512)\nnorm = nn.LayerNorm(512)\ny = norm(x)\n# B: Over last dimension of size 512 per token",
         "x = torch.randn(8, 10, 512)\nnorm = nn.LayerNorm(512)\ny = norm(x)\n# C: Across all 8*10*512 elements",
         "x = torch.randn(8, 10, 512)\nnorm = nn.LayerNorm(512)\ny = norm(x)\n# D: Only during training",
     ], "answer": 2},

    {"type": "code", "title": "requires_grad Check",
     "body": r"After this code, does \texttt{y.requires\_grad} equal \texttt{True}?",
     "code_options": [
         "x = torch.tensor([1.0], requires_grad=True)\ny = x * 2\n# y.requires_grad = False",
         "x = torch.tensor([1.0], requires_grad=True)\ny = x * 2\n# y.requires_grad = True",
         "x = torch.tensor([1.0], requires_grad=True)\ny = x * 2\n# y.requires_grad depends on x.dtype",
         "x = torch.tensor([1.0], requires_grad=True)\ny = x * 2\n# AttributeError: no requires_grad on y",
     ], "answer": 2},

    {"type": "code", "title": "Saving and Loading Model",
     "body": r"Which code correctly saves and reloads only model weights?",
     "code_options": [
         "torch.save(model, 'model.pth')\nmodel = torch.load('model.pth')",
         "torch.save(model.state_dict(), 'model.pth')\nmodel.load_state_dict(torch.load('model.pth'))",
         "model.save('model.pth')\nmodel = Model.load('model.pth')",
         "torch.export(model, 'model.pth')\nmodel = torch.import('model.pth')",
     ], "answer": 2},

    {"type": "code", "title": "Depthwise Conv in PyTorch",
     "body": r"Which argument makes \texttt{Conv2d} a depthwise convolution for $C=16$ channels?",
     "code_options": [
         "nn.Conv2d(16, 16, kernel_size=3, groups=1)",
         "nn.Conv2d(16, 16, kernel_size=3, groups=16)",
         "nn.Conv2d(16, 16, kernel_size=3, depthwise=True)",
         "nn.Conv2d(16, 1, kernel_size=3, groups=16)",
     ], "answer": 2},

    {"type": "code", "title": "Batch Dimension Check",
     "body": r"What is the value of \texttt{n}?",
     "code_options": [
         "x = torch.randn(16, 3, 224, 224)\nn = x.shape[0]\n# n = 3",
         "x = torch.randn(16, 3, 224, 224)\nn = x.shape[0]\n# n = 16",
         "x = torch.randn(16, 3, 224, 224)\nn = x.shape[0]\n# n = 224",
         "x = torch.randn(16, 3, 224, 224)\nn = x.shape[0]\n# n = 4",
     ], "answer": 2},

    {"type": "code", "title": "Softmax Axis",
     "body": r"For class probabilities from logits of shape $(B, C)$, which is correct?",
     "code_options": [
         "probs = F.softmax(logits, dim=0)  # softmax over batch",
         "probs = F.softmax(logits, dim=1)  # softmax over classes",
         "probs = F.softmax(logits, dim=-2) # softmax over batch",
         "probs = F.softmax(logits)         # default dim=None",
     ], "answer": 2},

    {"type": "code", "title": "torch.argmax Usage",
     "body": r"What does \texttt{pred} contain?",
     "code_options": [
         "logits = torch.tensor([[0.1, 2.0, 0.5]])\npred = torch.argmax(logits, dim=1)\n# pred = tensor([0])",
         "logits = torch.tensor([[0.1, 2.0, 0.5]])\npred = torch.argmax(logits, dim=1)\n# pred = tensor([1])",
         "logits = torch.tensor([[0.1, 2.0, 0.5]])\npred = torch.argmax(logits, dim=1)\n# pred = tensor([2.0])",
         "logits = torch.tensor([[0.1, 2.0, 0.5]])\npred = torch.argmax(logits, dim=1)\n# pred = tensor([3])",
     ], "answer": 2},

    {"type": "code", "title": "Bidirectional LSTM",
     "body": r"For a bidirectional LSTM with hidden size $h=32$, what is the output feature size per timestep?",
     "code_options": [
         "lstm = nn.LSTM(10, 32, bidirectional=True, batch_first=True)\n# output size per timestep = 32",
         "lstm = nn.LSTM(10, 32, bidirectional=True, batch_first=True)\n# output size per timestep = 64",
         "lstm = nn.LSTM(10, 32, bidirectional=True, batch_first=True)\n# output size per timestep = 16",
         "lstm = nn.LSTM(10, 32, bidirectional=True, batch_first=True)\n# output size per timestep = 128",
     ], "answer": 2},

    {"type": "code", "title": "Custom Activation",
     "body": r"Which correctly implements Swish activation $f(x) = x \cdot \sigma(x)$?",
     "code_options": [
         "def swish(x):\n    return x + torch.sigmoid(x)",
         "def swish(x):\n    return x * torch.sigmoid(x)",
         "def swish(x):\n    return torch.sigmoid(x * x)",
         "def swish(x):\n    return x * torch.tanh(x)",
     ], "answer": 2},

    {"type": "code", "title": "nn.Sequential Forward",
     "body": r"What is the output shape?",
     "code_options": [
         "model = nn.Sequential(\n    nn.Linear(128, 64),\n    nn.ReLU(),\n    nn.Linear(64, 10)\n)\nx = torch.randn(32, 128)\n# output shape = (32, 64)",
         "model = nn.Sequential(\n    nn.Linear(128, 64),\n    nn.ReLU(),\n    nn.Linear(64, 10)\n)\nx = torch.randn(32, 128)\n# output shape = (32, 10)",
         "model = nn.Sequential(\n    nn.Linear(128, 64),\n    nn.ReLU(),\n    nn.Linear(64, 10)\n)\nx = torch.randn(32, 128)\n# output shape = (32, 128)",
         "model = nn.Sequential(\n    nn.Linear(128, 64),\n    nn.ReLU(),\n    nn.Linear(64, 10)\n)\nx = torch.randn(32, 128)\n# output shape = (10, 32)",
     ], "answer": 2},

    {"type": "code", "title": "torch.stack vs torch.cat",
     "body": r"What is the shape of \texttt{out}?",
     "code_options": [
         "a = torch.randn(3, 4)\nb = torch.randn(3, 4)\nout = torch.stack([a, b], dim=0)\n# shape = (3, 8)",
         "a = torch.randn(3, 4)\nb = torch.randn(3, 4)\nout = torch.stack([a, b], dim=0)\n# shape = (2, 3, 4)",
         "a = torch.randn(3, 4)\nb = torch.randn(3, 4)\nout = torch.stack([a, b], dim=0)\n# shape = (6, 4)",
         "a = torch.randn(3, 4)\nb = torch.randn(3, 4)\nout = torch.stack([a, b], dim=0)\n# shape = (3, 4)",
     ], "answer": 2},

    {"type": "code", "title": "Parameter Groups in Optimizer",
     "body": r"What does using parameter groups in an optimizer allow?",
     "code_options": [
         "optimizer = torch.optim.SGD([\n    {'params': model.backbone.parameters(), 'lr': 1e-4},\n    {'params': model.head.parameters(), 'lr': 1e-3}\n], momentum=0.9)\n# A: Freeze backbone weights",
         "optimizer = torch.optim.SGD([\n    {'params': model.backbone.parameters(), 'lr': 1e-4},\n    {'params': model.head.parameters(), 'lr': 1e-3}\n], momentum=0.9)\n# B: Different learning rates for different parts",
         "optimizer = torch.optim.SGD([\n    {'params': model.backbone.parameters(), 'lr': 1e-4},\n    {'params': model.head.parameters(), 'lr': 1e-3}\n], momentum=0.9)\n# C: Train only head, ignore backbone",
         "optimizer = torch.optim.SGD([\n    {'params': model.backbone.parameters(), 'lr': 1e-4},\n    {'params': model.head.parameters(), 'lr': 1e-3}\n], momentum=0.9)\n# D: Use two optimizers simultaneously",
     ], "answer": 2},

    {"type": "code", "title": "Flatten Layer",
     "body": r"What is the shape of \texttt{y}?",
     "code_options": [
         "x = torch.randn(8, 64, 7, 7)\ny = x.flatten(1)\n# shape = (8, 64, 7, 7)",
         "x = torch.randn(8, 64, 7, 7)\ny = x.flatten(1)\n# shape = (8, 3136)",
         "x = torch.randn(8, 64, 7, 7)\ny = x.flatten(1)\n# shape = (8*64*7*7,)",
         "x = torch.randn(8, 64, 7, 7)\ny = x.flatten(1)\n# shape = (8, 64)",
     ], "answer": 2},

    {"type": "code", "title": "ReduceLROnPlateau",
     "body": r"When does \texttt{ReduceLROnPlateau} reduce the learning rate?",
     "code_options": [
         "scheduler = ReduceLROnPlateau(optimizer, patience=5)\n# A: Every 5 epochs unconditionally",
         "scheduler = ReduceLROnPlateau(optimizer, patience=5)\n# B: When monitored metric stops improving for 5 epochs",
         "scheduler = ReduceLROnPlateau(optimizer, patience=5)\n# C: When loss increases for 1 epoch",
         "scheduler = ReduceLROnPlateau(optimizer, patience=5)\n# D: When accuracy exceeds 95%",
     ], "answer": 2},
]


random.seed(123)
random.shuffle(MCQS)
MCQS = MCQS[:100]


def make_latex(idx, mcq):
    is_code = mcq["type"] == "code"
    letters = ["A", "B", "C", "D"]

    if is_code:
        opts_latex = ""
        for i, opt in enumerate(mcq["code_options"]):
            opts_latex += f"\n\\noindent\\textbf{{{letters[i]}.}}\n"
            opts_latex += f"\\begin{{lstlisting}}\n{opt}\n\\end{{lstlisting}}\n"
            opts_latex += "\\vspace{0.3em}\n"
    else:
        opts_latex = "\\begin{enumerate}[label=\\Alph*.]\n"
        for opt in mcq["options"]:
            opts_latex += f"  \\item {opt}\n"
        opts_latex += "\\end{enumerate}\n"

    body = mcq["body"]
    if mcq["type"] == "equation":
        body = body.replace("$$", "\n$$\n")

    tex = r"""\documentclass[12pt]{article}
\usepackage[margin=1.5cm, top=1.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{parskip}
\usepackage{lmodern}
\usepackage[T1]{fontenc}

\lstset{
  basicstyle=\ttfamily\footnotesize,
  backgroundcolor=\color{gray!10},
  frame=single,
  framesep=4pt,
  breaklines=true,
  columns=fullflexible,
  keepspaces=true
}

\begin{document}
\pagestyle{empty}

{\large\bfseries Ques: """ + mcq["title"] + r"""}

\vspace{0.5em}
""" + body + r"""

\vspace{1em}
{\bfseries Options}

""" + opts_latex + r"""

\end{document}
"""
    return tex


def compile_latex_to_png(idx, tex_str, out_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "question.tex")
        pdf_path = os.path.join(tmpdir, "question.pdf")

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_str)

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "question.tex"],
            cwd=tmpdir, capture_output=True, text=True,
        )

        if not os.path.exists(pdf_path):
            print(f"  [ERROR] pdflatex failed for image_{idx}")
            print(result.stdout[-300:])
            return None

        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=1)
        out_path = os.path.join(out_dir, f"image_{idx}.png")
        images[0].save(out_path, "PNG")
        return out_path


def main():
    out_dir = "synthetic_test3/images"
    os.makedirs(out_dir, exist_ok=True)

    with open("synthetic_test3/test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name"])
        for i in range(1, len(MCQS) + 1):
            writer.writerow([f"image_{i}"])

    with open("synthetic_test3/ground_truth.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "correct_option"])
        for i, mcq in enumerate(MCQS, 1):
            writer.writerow([f"image_{i}", mcq["answer"]])

    print(f"Generating {len(MCQS)} MCQ images -> synthetic_test3/")
    for i, mcq in enumerate(MCQS, 1):
        tex = make_latex(i, mcq)
        path = compile_latex_to_png(i, tex, out_dir)
        status = "OK" if path else "FAILED"
        print(f"  [{i:03d}] {mcq['type']:12s} [{status}]  answer={mcq['answer']}")

    print(f"\nDone! Run: python inference.py --test_dir synthetic_test3")


if __name__ == "__main__":
    main()
