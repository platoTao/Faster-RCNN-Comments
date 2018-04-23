"""A `MutableModule` implement the `BaseModule` API, and allows input shape
varying with training iterations. If shapes vary, executors will rebind,
如果形状不一样，就会rebind（符号式）
using shared arrays from the initial module binded with maximum shape.
"""

import logging

#symbol 神经网络的定义
#context：执行运算的设备
#data_names：网络输入数据的列表
#label_names：网络输入标签的列表

from mxnet import context as ctx
from mxnet.initializer import Uniform
from mxnet.module.base_module import BaseModule
from mxnet.module.module import Module

class MutableModule(BaseModule):
    """A mutable module is a module that supports variable input data.
        支持不同形状的输入数据
    Parameters
    ----------
    symbol : Symbol（符号式，类似于tensorflow的placeholder）
    data_names : list of str
    label_names : list of str
    logger : Logger
    context : Context or list of Context
    work_load_list : list of number
    max_data_shapes : list of (name, shape) tuple, designating（指定） inputs whose shape vary
    max_label_shapes : list of (name, shape) tuple, designating inputs whose shape vary
    fixed_param_prefix : list of str, indicating fixed parameters
    (faster-RCNN训练过程分四个步骤，其中3,4阶段会固定住某些层的参数)
    """
    def __init__(self, symbol, data_names, label_names,
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 max_data_shapes=None, max_label_shapes=None, fixed_param_prefix=None):
        super(MutableModule, self).__init__(logger=logger)
        self._symbol = symbol
        self._data_names = data_names
        self._label_names = label_names
        self._context = context
        self._work_load_list = work_load_list

        self._curr_module = None
        self._max_data_shapes = max_data_shapes
        self._max_label_shapes = max_label_shapes
        self._fixed_param_prefix = fixed_param_prefix

        #需要固定住参数的网络层名字
        fixed_param_names = list()
        if fixed_param_prefix is not None:
            #list_arguments()：Lists all the arguments in the symbol.
            for name in self._symbol.list_arguments():
                for prefix in self._fixed_param_prefix:
                    if prefix in name:
                        fixed_param_names.append(name)
        self._fixed_param_names = fixed_param_names

    def _reset_bind(self):
        self.binded = False
        self._curr_module = None

    @property
    def data_names(self):
        return self._data_names

    @property
    def output_names(self):
        #Lists all the outputs in the symbol.
        return self._symbol.list_outputs()

    @property
    def data_shapes(self):
        assert self.binded
        return self._curr_module.data_shapes

    @property
    def label_shapes(self):
        assert self.binded
        return self._curr_module.label_shapes

    @property
    def output_shapes(self):
        assert self.binded
        return self._curr_module.output_shapes

    def get_params(self):
        assert self.binded and self.params_initialized
        return self._curr_module.get_params()
    #
    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False):
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'
        self._curr_module.init_params(initializer=initializer, arg_params=arg_params,
                                      aux_params=aux_params, allow_missing=allow_missing,
                                      force_init=force_init)
        self.params_initialized = True

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None):
        # in case we already initialized params, keep it
        if self.params_initialized:
            arg_params, aux_params = self.get_params()

        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        assert shared_module is None, 'shared_module for MutableModule is not supported'

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True

        max_shapes_dict = dict()
        if self._max_data_shapes is not None:
            # dict1.update(dict2) 用于把 字典2中的键值对更新到 字典1 中
            max_shapes_dict.update(dict(self._max_data_shapes))
        if self._max_label_shapes is not None:
            max_shapes_dict.update(dict(self._max_label_shapes))

        max_data_shapes = list()
        for name, shape in data_shapes:
            if name in max_shapes_dict:
                max_data_shapes.append((name, max_shapes_dict[name]))
            else:
                max_data_shapes.append((name, shape))

        max_label_shapes = list()
        if label_shapes is not None:
            for name, shape in label_shapes:
                if name in max_shapes_dict:
                    max_label_shapes.append((name, max_shapes_dict[name]))
                else:
                    max_label_shapes.append((name, shape))

        if len(max_label_shapes) == 0:
            max_label_shapes = None
        # referen： https://mxnet.incubator.apache.org/api/python/module/module.html#mxnet.module.Module
        # 初始化 模型
        module = Module(self._symbol, self._data_names, self._label_names, logger=self.logger,
                        context=self._context, work_load_list=self._work_load_list,
                        fixed_param_names=self._fixed_param_names)
        module.bind(max_data_shapes, max_label_shapes, for_training, inputs_need_grad,
                    force_rebind=False, shared_module=None)
        self._curr_module = module

        # copy back saved params, if already initialized
        if self.params_initialized:
            self.set_params(arg_params, aux_params)
    #初始化优化器
    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return
        #初始化优化器
        self._curr_module.init_optimizer(kvstore, optimizer, optimizer_params,
                                         force_init=force_init)
        self.optimizer_initialized = True

    def forward(self, data_batch, is_train=None):
        assert self.binded and self.params_initialized

        # get current_shapes
        if self._curr_module.label_shapes is not None:
            current_shapes = dict(self._curr_module.data_shapes + self._curr_module.label_shapes)
        else:
            current_shapes = dict(self._curr_module.data_shapes)

        # get input_shapes
        if data_batch.provide_label is not None:
            input_shapes = dict(data_batch.provide_data + data_batch.provide_label)
        else:
            input_shapes = dict(data_batch.provide_data)

        # decide if shape changed
        shape_changed = False
        for k, v in current_shapes.items():
            if v != input_shapes[k]:
                shape_changed = True

        if shape_changed:
            module = Module(self._symbol, self._data_names, self._label_names,
                            logger=self.logger, context=self._context,
                            work_load_list=self._work_load_list,
                            fixed_param_names=self._fixed_param_names)
            module.bind(data_batch.provide_data, data_batch.provide_label, self._curr_module.for_training,
                        self._curr_module.inputs_need_grad, force_rebind=False,
                        shared_module=self._curr_module)
            self._curr_module = module

        self._curr_module.forward(data_batch, is_train=is_train)

    def backward(self, out_grads=None):
        assert self.binded and self.params_initialized
        self._curr_module.backward(out_grads=out_grads)
    # 根据优化器和上次前向传播的梯度计算结果来更新参数
    # Updates parameters according to the installed optimizer and the gradients computed in the previous forward-backward batch.
    def update(self):
        assert self.binded and self.params_initialized and self.optimizer_initialized
        self._curr_module.update()

    def get_outputs(self, merge_multi_context=True):
        assert self.binded and self.params_initialized
        return self._curr_module.get_outputs(merge_multi_context=merge_multi_context)
    # Gets the gradients to the inputs, computed in the previous backward computation.
    # 获得前次反向传播计算出来的梯度 ，送入到输入
    def get_input_grads(self, merge_multi_context=True):
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._curr_module.get_input_grads(merge_multi_context=merge_multi_context)
    # Evaluates and accumulates evaluation metric on outputs of the last forward computation.
    # 根据上一次的forward结果，更新评价指标
    def update_metric(self, eval_metric, labels):
        assert self.binded and self.params_initialized
        self._curr_module.update_metric(eval_metric, labels)

    def install_monitor(self, mon):
        """ Install monitor on all executors """
        assert self.binded
        self._curr_module.install_monitor(mon)
