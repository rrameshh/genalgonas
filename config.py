# config.py

NUM_LAYERS = 3
CLASSES = 10
DATASET = 'mnist'
POPULATION_SIZE = 3
NUM_GENERATIONS = 2
MUTATION_RATE = 0.1

# Spaces definition
LAYER_TYPE_SPACE = ["conv", "maxpool", "depthwise_conv", "separable_conv", "avgpool", "global_avgpool", "identity"]
KERNEL_SIZE_SPACE = [3, 5, 7]
STRIDE_SPACE = [1, 2]
FILTERS_SPACE = [16, 32, 64, 128, 256]
RESIDUAL_SPACE = ["None", "Add", "Concatenate"]
NORMALIZATION_SPACE = ["BatchNorm", "LayerNorm"]
ACTIVATION_SPACE = ["relu", "relu6", "silu", "swish"]
POOLING_TYPE_SPACE = ["max", "avg", "adaptive_avg"]
