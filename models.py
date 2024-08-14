# models.py

import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, DepthwiseConv2D, SeparableConv2D,
                                     GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, LayerNormalization,
                                     Activation, Flatten, Dense, Add, Concatenate)

def model_to_json(model):
    json_model = {"layers": []}
    for layer in model.layers:
        layer_config = {}
        if isinstance(layer, Conv2D):
            layer_config['layer_type'] = 'conv'
            layer_config['filters'] = layer.filters
            layer_config['kernel_size'] = layer.kernel_size
            layer_config['activation'] = layer.activation.__name__
        elif isinstance(layer, MaxPooling2D):
            layer_config['layer_type'] = 'maxpool'
            layer_config['pool_size'] = layer.pool_size
        elif isinstance(layer, DepthwiseConv2D):
            layer_config['layer_type'] = 'depthwise_conv'
            layer_config['kernel_size'] = layer.kernel_size
            layer_config['activation'] = layer.activation.__name__
        elif isinstance(layer, SeparableConv2D):
            layer_config['layer_type'] = 'separable_conv'
            layer_config['filters'] = layer.filters
            layer_config['kernel_size'] = layer.kernel_size
            layer_config['activation'] = layer.activation.__name__
        elif isinstance(layer, AveragePooling2D):
            layer_config['layer_type'] = 'avgpool'
            layer_config['pool_size'] = layer.pool_size
        elif isinstance(layer, GlobalAveragePooling2D):
            layer_config['layer_type'] = 'global_avgpool'
        elif isinstance(layer, BatchNormalization):
            layer_config['layer_type'] = 'batchnorm'
        elif isinstance(layer, LayerNormalization):
            layer_config['layer_type'] = 'layernorm'
        elif isinstance(layer, Add):
            layer_config['layer_type'] = 'add'
        elif isinstance(layer, Concatenate):
            layer_config['layer_type'] = 'concatenate'
        elif isinstance(layer, Activation):
            layer_config['layer_type'] = 'activation'
            layer_config['activation'] = layer.activation.__name__
        elif isinstance(layer, Dense):
            layer_config['layer_type'] = 'dense'
            layer_config['units'] = layer.units
            layer_config['activation'] = layer.activation.__name__
        elif isinstance(layer, Flatten):
            layer_config['layer_type'] = 'flatten'

        json_model['layers'].append(layer_config)
    return json.dumps(json_model)

def json_to_model(json_str):
    model_config = json.loads(json_str)
    model = Sequential()
    for layer_config in model_config['layers']:
        if layer_config['layer_type'] == 'conv':
            model.add(Conv2D(filters=layer_config['filters'], kernel_size=layer_config['kernel_size'],
                             activation=layer_config['activation'], input_shape=(28, 28, 1)))
        elif layer_config['layer_type'] == 'maxpool':
            model.add(MaxPooling2D(pool_size=layer_config['pool_size']))
        elif layer_config['layer_type'] == 'depthwise_conv':
            model.add(DepthwiseConv2D(kernel_size=layer_config['kernel_size'], activation=layer_config['activation']))
        elif layer_config['layer_type'] == 'separable_conv':
            model.add(SeparableConv2D(filters=layer_config['filters'], kernel_size=layer_config['kernel_size'],
                                      activation=layer_config['activation']))
        elif layer_config['layer_type'] == 'avgpool':
            model.add(AveragePooling2D(pool_size=layer_config['pool_size']))
        elif layer_config['layer_type'] == 'global_avgpool':
            model.add(GlobalAveragePooling2D())
        elif layer_config['layer_type'] == 'batchnorm':
            model.add(BatchNormalization())
        elif layer_config['layer_type'] == 'layernorm':
            model.add(LayerNormalization())
        elif layer_config['layer_type'] == 'add':
            model.add(Add())
        elif layer_config['layer_type'] == 'concatenate':
            model.add(Concatenate())
        elif layer_config['layer_type'] == 'activation':
            model.add(Activation(layer_config['activation']))
        elif layer_config['layer_type'] == 'dense':
            model.add(Dense(units=layer_config['units'], activation=layer_config['activation']))
        elif layer_config['layer_type'] == 'flatten':
            model.add(Flatten())

    return model
