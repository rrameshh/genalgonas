# utils.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, DepthwiseConv2D, SeparableConv2D,
                                     GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, LayerNormalization,
                                     Activation, Flatten, Dense, Add, Concatenate)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import time

def evaluate_architecture(architecture, weights=False, dataset='mnist'):
    model = Sequential()
    layer_outputs = []
    input_shape_set = False

    for config in architecture:
        layer_type = config['layer_type']

        if layer_type == 'conv':
            if not input_shape_set:
                input_shape = (28, 28, 1) if dataset == 'mnist' else (32, 32, 3)
                model.add(Conv2D(filters=config.get('filters', 32), kernel_size=config.get('kernel_size', (3, 3)),
                                 activation=config.get('activation', 'relu'), input_shape=input_shape))
                input_shape_set = True
            else:
                model.add(Conv2D(filters=config.get('filters', 32), kernel_size=config.get('kernel_size', (3, 3)),
                                 activation=config.get('activation', 'relu')))
        elif layer_type == 'maxpool':
            if len(layer_outputs) != 0 and model.layers[-1].output_shape[1] >= 5 and model.layers[-1].output_shape[2] >= 5:
                model.add(MaxPooling2D(pool_size=config.get('pool_size', (2, 2))))
        elif layer_type == 'depthwise_conv':
            if len(layer_outputs) != 0 and model.layers[-1].output_shape[1] >= 5 and model.layers[-1].output_shape[2] >= 5:
                model.add(DepthwiseConv2D(kernel_size=config.get('kernel_size', (3, 3)), activation=config.get('activation', 'relu')))
        elif layer_type == 'separable_conv':
            model.add(SeparableConv2D(filters=config.get('filters', 32), kernel_size=config.get('kernel_size', (3, 3)), activation=config.get('activation', 'relu')))
        elif layer_type == 'avgpool':
            if len(layer_outputs) != 0 and model.layers[-1].output_shape[1] >= 5 and model.layers[-1].output_shape[2] >= 5:
                model.add(AveragePooling2D(pool_size=config.get('pool_size', (2, 2))))
        elif layer_type == 'global_avgpool':
            if len(model.layers) >= 2 and isinstance(model.layers[-1], Dense) and isinstance(model.layers[-2], Flatten):
                pass
            else:
                if len(layer_outputs) != 0 and model.layers[-1].output_shape[1] >= 5 and model.layers[-1].output_shape[2] >= 5:
                    model.add(GlobalAveragePooling2D())
        elif layer_type == 'identity':
            pass

        normalization_type = config.get('normalization', None)
        if normalization_type == 'BatchNorm':
            if len(layer_outputs) == 0:
                config['normalization'] = None
            else:
                model.add(BatchNormalization())
        elif normalization_type == 'LayerNorm':
            if len(layer_outputs) == 0:
                config['normalization'] = None
            else:
                model.add(LayerNormalization())

        residual_type = config.get('residual', None)
        if residual_type == 'Add':
            if layer_outputs:
                if len(layer_outputs) > 1:
                    added = Add()(layer_outputs)
                else:
                    added = layer_outputs[0]
                model.add(Conv2D(filters=config.get('filters', 32), kernel_size=(1, 1), activation='relu'))
                model.add(added)
            layer_outputs = []

        activation_type = config.get('activation', None)
        if activation_type:
            model.add(Activation(activation_type))

        if len(layer_outputs) != 0:
            layer_outputs.append(model.layers[-1].output)

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train / 255.0, axis=-1)
        x_test = np.expand_dims(x_test / 255.0, axis=-1)
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0

    y_train = to_categorical(y_train, num_classes=CLASSES)
    y_test = to_categorical(y_test, num_classes=CLASSES)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    start_time = time.time()
    model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping], verbose=0)
    latency = time.time() - start_time

    model_size = get_model_memory(model, batch_size=1)
    flops = get_flops(model, None)

    if weights:
        return model, model.get_weights()

    _, accuracy = model.evaluate(x_test, y_test)

    return accuracy, latency, model_size, flops

def get_model_memory(model, batch_size):
    features_mem = 0
    float_bytes = 4.0
    single_layer_mem_float = 0

    for layer in model.layers:
        out_shape = layer.output.shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        else:
            if len(out_shape) >= 4:
                out_shape = [out_shape[1], out_shape[2], out_shape[3]]
            else:
                out_shape = [out_shape[1]]

        single_layer_mem = 1
        for s in out_shape:
            if s is not None:
                single_layer_mem *= s
        single_layer_mem_float += single_layer_mem * float_bytes
        single_layer_mem_mb = single_layer_mem_float / (1024 * 1024)
        features_mem += single_layer_mem_mb

    trainable_wts = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_wts = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    parameter_mem_MB = ((trainable_wts + non_trainable_wts) * float_bytes) / (1024**2)
    total_memory_MB = (batch_size * features_mem) + parameter_mem_MB
    total_memory_GB = total_memory_MB / 1024

    return total_memory_GB

def get_flops(model, batch_size=None):
    if batch_size is None:
        batch_size = 1

    input_shape = list(model.inputs[0].shape[1:])
    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + input_shape, model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                          run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops
