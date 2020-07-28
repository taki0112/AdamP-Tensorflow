## AdamP Optimizer &mdash; TensorFlow Implementation
### [[Paper]](https://arxiv.org/abs/2006.08217) [[Project page]](https://clovaai.github.io/AdamP/) [[Pytorch]](https://github.com/clovaai/AdamP)

<div align="center">
  <img src=https://clovaai.github.io/AdamP/static/img/projection.svg>
  <img src=https://clovaai.github.io/AdamP/static/img/algorithms.svg height=400px>
</div>

## Usage
Usage is exactly same as [tf.keras.optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) library!
```python

from adamp_tf import AdamP
from sgdp_tf import SGDP

optimizer_adamp = AdamP(learning_rate=0.001, betas=(0.9, 0.999), weight_decay=1e-2)
optimizer_sgdp = SGDP(learning_rate=0.1, weight_decay=1e-5, momentum=0.9, nesterov=True)

```

## Arguments
`SGDP` and `AdamP` share arguments with [tf.keras.optimizers.SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) and [tf.keras.optimizers.Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam).
There are two additional hyperparameters; we recommend using the default values.
- `delta` : threhold that determines whether a set of parameters is scale invariant or not (default: 0.1)
- `wd_ratio` : relative weight decay applied on _scale-invariant_ parameters compared to that applied on _scale-variant_ parameters (default: 0.1)

Both `SGDP` and `AdamP` support Nesterov momentum.
- `nesterov` : enables Nesterov momentum (default: False)

## Experimental results
### ImageNet classificaiton
Accuracies of state-of-the-art networks ([MobileNetV2](https://arxiv.org/abs/1801.04381), [ResNet](https://arxiv.org/abs/1512.03385), and [CutMix-ed ResNet](https://arxiv.org/abs/1905.04899)) trained with SGDP and AdamP.
<div align="center">
  <img src=https://clovaai.github.io/AdamP/static/img/table01.svg height=160px>
</div>

## MS-COCO object detection
Average precision (AP) scores of [CenterNet](https://arxiv.org/abs/1904.07850) and [SSD](https://arxiv.org/abs/1512.02325) trained with Adam and AdamP optimizers.
<div align="center">
  <img src=https://clovaai.github.io/AdamP/static/img/table03.svg height=160px>
</div>

## Adversarial training
Standard accuracies and attacked accuracies of [Wide-ResNet trained on CIFAR-10 with PGD-10 attacks](https://github.com/louis2889184/pytorch-adversarial-training).
<div align="center">
  <img src=https://clovaai.github.io/AdamP/static/img/table04_0.svg height=160px>
</div>

## Robustness against real-world biases (Biased-MNIST)
Unbiased accuraccy with [ReBias](https://arxiv.org/abs/1910.02806).
<div align="center">
  <img src=https://clovaai.github.io/AdamP/static/img/table04_1.svg height=160px>
</div>

## Robustness against real-world biases (9-Class ImageNet)
Biased / unbiased / [ImageNet-A](https://arxiv.org/abs/1907.07174) accuraccy with[ReBias](https://arxiv.org/abs/1910.02806).
<div align="center">
  <img src=https://clovaai.github.io/AdamP/static/img/table05.svg height=160px>
</div>

## Audio classification
Results on three audio classification tasks with [Harmonic CNN](https://ccrma.stanford.edu/~urinieto/MARL/publications/ICASSP2020_Won.pdf).
<div align="center">
  <img src=https://clovaai.github.io/AdamP/static/img/table05.svg height=160px>
</div>

## Image retrieval
Recall@1 on CUB, Cars-196, InShop, and SOP datasets. ImageNet-pretrained ResNet50 networks are fine-tuned by the triplet (semi-hard mining) and the [ProxyAnchor (PA) loss](https://arxiv.org/abs/2003.13911).
<div align="center">
  <img src=https://clovaai.github.io/AdamP/static/img/table06.svg height=180px>
</div>

## Author
[Junho Kim](http://bit.ly/jhkim_ai)
