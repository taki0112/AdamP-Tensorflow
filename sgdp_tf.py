""" SGDP for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops


class SGDP(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True
    def __init__(self,
                 learning_rate=0.1,
                 momentum=0.0,
                 dampening=0.0,
                 weight_decay=0.0,
                 nesterov=False,
                 epsilon=1e-8,
                 delta=0.1,
                 wd_ratio=0.1,
                 name="SGDP",
                 **kwargs):

        super(SGDP, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")

        self._set_hyper("momentum", momentum)
        self._set_hyper("dampening", dampening)
        self._set_hyper("epsilon", epsilon)
        self._set_hyper("delta", delta)
        self._set_hyper("wd_ratio", wd_ratio)

        self.nesterov = nesterov
        self.weight_decay = weight_decay

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")
        for var in var_list:
            self.add_slot(var, "buf")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(SGDP, self)._prepare_local(var_device, var_dtype, apply_state)
        lr = apply_state[(var_device, var_dtype)]['lr_t']

        momentum = array_ops.identity(self._get_hyper("momentum", var_dtype))
        dampening = array_ops.identity(self._get_hyper('dampening', var_dtype))
        delta = array_ops.identity(self._get_hyper('delta', var_dtype))
        wd_ratio = array_ops.identity(self._get_hyper('wd_ratio', var_dtype))

        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                weight_decay=ops.convert_to_tensor_v2(self.weight_decay, var_dtype),
                momentum=momentum,
                dampening=dampening,
                delta=delta,
                wd_ratio=wd_ratio))


    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        buf = self.get_slot(var, 'buf')
        b_scaled_g_values = grad * (1 - coefficients['dampening'])
        buf_t = state_ops.assign(buf, buf * coefficients['momentum'] + b_scaled_g_values, use_locking=self._use_locking)

        if self.nesterov:
            d_p = grad + coefficients['momentum'] * buf_t
        else:
            d_p = buf_t

        # Projection
        wd_ratio = 1
        if len(var.shape) > 1:
            d_p, wd_ratio = self._projection(var, grad, d_p, coefficients['delta'], coefficients['wd_ratio'], coefficients['epsilon'])

        # Weight decay
        if self.weight_decay > 0:
            var = state_ops.assign(var, var * (1 - coefficients['lr'] * coefficients['weight_decay'] * wd_ratio), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, coefficients['lr'] * d_p, use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, buf_t])

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # This method is only needed for momentum optimization.
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        buf = self.get_slot(var, 'buf')
        b_scaled_g_values = grad * (1 - coefficients['dampening'])
        buf_t = state_ops.assign(buf, buf * coefficients['momentum'], use_locking=self._use_locking)

        with ops.control_dependencies([buf_t]):
            buf_t = self._resource_scatter_add(buf, indices, b_scaled_g_values)

        if self.nesterov:
            d_p = self._resource_scatter_add(buf_t * coefficients['momentum'], indices, grad)
        else:
            d_p = buf_t

        # Projection
        wd_ratio = 1
        if len(array_ops.shape(var)) > 1:
            d_p, wd_ratio = self._projection(var, grad, d_p, coefficients['delta'], coefficients['wd_ratio'],
                                             coefficients['epsilon'])

        # Weight decay
        if self.weight_decay > 0:
            var = state_ops.assign(var, var * (1 - coefficients['lr'] * coefficients['weight_decay'] * wd_ratio), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, coefficients['lr'] * d_p, use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, buf_t])

    def _channel_view(self, x):
        return array_ops.reshape(x, shape=[array_ops.shape(x)[0], -1])

    def _layer_view(self, x):
        return array_ops.reshape(x, shape=[1, -1])

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        x_norm = math_ops.euclidean_norm(x, axis=-1) + eps
        y_norm = math_ops.euclidean_norm(y, axis=-1) + eps
        dot = math_ops.reduce_sum(x * y, axis=-1)

        return math_ops.abs(dot) / x_norm / y_norm

    def _projection(self, var, grad, perturb, delta, wd_ratio, eps):
        # channel_view
        cosine_sim = self._cosine_similarity(grad, var, eps, self._channel_view)
        cosine_max = math_ops.reduce_max(cosine_sim)
        compare_val = delta / math_ops.sqrt(math_ops.cast(self._channel_view(var).shape[-1], dtype=delta.dtype))

        perturb, wd = control_flow_ops.cond(pred=cosine_max < compare_val,
                                            true_fn=lambda : self.channel_true_fn(var, perturb, wd_ratio, eps),
                                            false_fn=lambda : self.channel_false_fn(var, grad, perturb, delta, wd_ratio, eps))

        return perturb, wd

    def channel_true_fn(self, var, perturb, wd_ratio, eps):
        expand_size = [-1] + [1] * (len(var.shape) - 1)
        var_n = var / (array_ops.reshape(math_ops.euclidean_norm(self._channel_view(var), axis=-1), shape=expand_size) + eps)
        perturb = state_ops.assign_sub(perturb, var_n * array_ops.reshape(math_ops.reduce_sum(self._channel_view(var_n * perturb), axis=-1), shape=expand_size))
        wd = wd_ratio

        return perturb, wd

    def channel_false_fn(self, var, grad, perturb, delta, wd_ratio, eps):
        cosine_sim = self._cosine_similarity(grad, var, eps, self._layer_view)
        cosine_max = math_ops.reduce_max(cosine_sim)
        compare_val = delta / math_ops.sqrt(math_ops.cast(self._layer_view(var).shape[-1], dtype=delta.dtype))

        perturb, wd = control_flow_ops.cond(cosine_max < compare_val,
                                              true_fn=lambda : self.layer_true_fn(var, perturb, wd_ratio, eps),
                                              false_fn=lambda : self.identity_fn(perturb))

        return perturb, wd

    def layer_true_fn(self, var, perturb, wd_ratio, eps):
        expand_size = [-1] + [1] * (len(var.shape) - 1)
        var_n = var / (array_ops.reshape(math_ops.euclidean_norm(self._layer_view(var), axis=-1), shape=expand_size) + eps)
        perturb = state_ops.assign_sub(perturb, var_n * array_ops.reshape(math_ops.reduce_sum(self._layer_view(var_n * perturb), axis=-1), shape=expand_size))
        wd = wd_ratio

        return perturb, wd

    def identity_fn(self, perturb):
        wd = 1.0

        return perturb, wd

    def get_config(self):
        config = super(SGDP, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'momentum': self._serialize_hyperparameter('momentum'),
            'dampening': self._serialize_hyperparameter('dampening'),
            'delta': self._serialize_hyperparameter('delta'),
            'wd_ratio': self._serialize_hyperparameter('wd_ratio'),
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            "nesterov": self.nesterov,
        })
        return config
