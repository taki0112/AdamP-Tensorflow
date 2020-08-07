## AdamP Optimizer &mdash; Unofficial TensorFlow Implementation
### "Slowing Down the Weight Norm Increase in Momentum-based Optimizers"
## Implemented by [Junho Kim](http://bit.ly/jhkim_ai)
### [[Paper]](https://arxiv.org/abs/2006.08217) [[Project page]](https://clovaai.github.io/AdamP/) [[Official Pytorch]](https://github.com/clovaai/AdamP)

<div align="center">
  <img src=https://clovaai.github.io/AdamP/static/img/projection.svg>
  <img src=https://clovaai.github.io/AdamP/static/img/algorithms.svg height=400px>
</div>


## Validation
I have checked that the code is working, but I couldn't confirm if the performance is the same as the offical code.

## Usage
Usage is exactly same as [tf.keras.optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) library!
```python

from adamp_tf import AdamP
from sgdp_tf import SGDP

optimizer_adamp = AdamP(learning_rate=0.001, betas=(0.9, 0.999), weight_decay=1e-2)
optimizer_sgdp = SGDP(learning_rate=0.1, weight_decay=1e-5, momentum=0.9, nesterov=True)
```
* **Do not use with `tf.nn.scale_regularization_loss`.** Use the `weight_decay` argument.

## Arguments
`SGDP` and `AdamP` share arguments with [tf.keras.optimizers.SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) and [tf.keras.optimizers.Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam).
There are two additional hyperparameters; we recommend using the default values.
- `delta` : threshold that determines whether a set of parameters is scale invariant or not (default: 0.1)
- `wd_ratio` : relative weight decay applied on _scale-invariant_ parameters compared to that applied on _scale-variant_ parameters (default: 0.1)

Both `SGDP` and `AdamP` support Nesterov momentum.
- `nesterov` : enables Nesterov momentum (default: False)

## How to cite

```
@article{heo2020adamp,
    title={Slowing Down the Weight Norm Increase in Momentum-based Optimizers},
    author={Heo, Byeongho and Chun, Sanghyuk and Oh, Seong Joon and Han, Dongyoon and Yun, Sangdoo and Uh, Youngjung and Ha, Jung-Woo},
    year={2020},
    journal={arXiv preprint arXiv:2006.08217},
}
```
