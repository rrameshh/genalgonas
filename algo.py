import random
import json
import numpy as np
import tensorflow as tf
import os
import time

from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, DepthwiseConv2D, SeparableConv2D, GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, LayerNormalization, Activation, Concatenate, Add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph




NUM_LAYERS = 3
CLASSES = 10
DATASET = 'mnist'
POPULATION_SIZE = 3
NUM_GENERATIONS = 2
MUTATION_RATE = 0.1


# Spaces definition
layer_type_space = ["conv", "maxpool", "depthwise_conv", "separable_conv", "avgpool", "global_avgpool", "identity"]
kernel_size_space = [3, 5, 7]  # Common kernel sizes for convolutional layers
stride_space = [1, 2]  # Stride options for convolutional and pooling layers
filters_space = [16, 32, 64, 128, 256]  # Common filter sizes for convolutional layers
residual_space = ["None", "Add", "Concatenate"]  # Options for residual connections
normalization_space = ["BatchNorm", "LayerNorm"]  # Common normalization layers
activation_space = ["relu", "relu6", "silu", "swish"]  # Activation functions
pooling_type_space = ["max", "avg", "adaptive_avg"]  # Types of pooling layers


# Example of a fitness function (evaluation metric)
def evaluate_architecture(architecture, weights, dataset=DATASET):
    model = Sequential()
    # inputs = Input(shape=(28, 28, 1))  # Assuming MNIST image shape
    layer_outputs = []  # To store outputs for potential concatenation or residual connection

    input_shape_set = False

    for config in architecture:
        layer_type = config['layer_type']

        if layer_type == 'conv':
            if not input_shape_set:
                if dataset == 'mnist':
                    input_shape = (28, 28, 1)
                elif dataset == 'cifar10':
                    input_shape = (32, 32, 3)
                model.add(Conv2D(filters=config.get('filters', 32), kernel_size=config.get('kernel_size', (3, 3)),
                                 activation=config.get('activation', 'relu'), input_shape=input_shape))
                input_shape_set = True
            else:
                model.add(Conv2D(filters=config.get('filters', 32), kernel_size=config.get('kernel_size', (3, 3)),
                                 activation=config.get('activation', 'relu')))
        elif layer_type == 'maxpool':
            # Ensure input dimensions are sufficient for pooling
            if len(layer_outputs) != 0 and model.layers[-1].output_shape[1] >= 5 and model.layers[-1].output_shape[2] >= 5:
                model.add(MaxPooling2D(pool_size=config.get('pool_size', (2, 2))))
            else:
                # Skip pooling if input dimensions are too small
                pass
        elif layer_type == 'depthwise_conv':
            if len(layer_outputs) != 0 and model.layers[-1].output_shape[1] >= 5 and model.layers[-1].output_shape[2] >= 5 and isinstance(model.layers[-1], Conv2D):
                model.add(DepthwiseConv2D(kernel_size=config.get('kernel_size', (3, 3)), activation=config.get('activation', 'relu')))
            # model.add(DepthwiseConv2D(kernel_size=config.get('kernel_size', (3, 3)), activation=config.get('activation', 'relu')))
        elif layer_type == 'separable_conv':
            model.add(SeparableConv2D(filters=config.get('filters', 32), kernel_size=config.get('kernel_size', (3, 3)), activation=config.get('activation', 'relu')))
        elif layer_type == 'avgpool':
            if len(layer_outputs) != 0 and model.layers[-1].output_shape[1] >= 5 and model.layers[-1].output_shape[2] >= 5:
                model.add(AveragePooling2D(pool_size=config.get('pool_size', (2, 2))))
            else:
                # Skip pooling if input dimensions are too small
                pass
        elif layer_type == 'global_avgpool':
        # Check if the last two layers are Flatten and Dense
          if len(model.layers) >= 2 and \
            isinstance(model.layers[-1], Dense) and \
            isinstance(model.layers[-2], Flatten):
              pass  # Skip adding GlobalAveragePooling2D
          else:
             # Ensure input dimensions are sufficient for pooling
              if len(layer_outputs) != 0 and model.layers[-1].output_shape[1] >= 5 and model.layers[-1].output_shape[2] >= 5:
                  model.add(GlobalAveragePooling2D())
        elif layer_type == 'identity':
            # Identity layer does nothing, so we skip adding any layer
            pass

        # Add normalization layer if specified
        normalization_type = config.get('normalization', None)
        if normalization_type == 'BatchNorm':
            if len(layer_outputs) == 0:
                #raise ValueError("BatchNormalization layer must have a preceding layer.")
                # model.add(None)
                config['normalization'] = None
            else:
                model.add(BatchNormalization())
        elif normalization_type == 'LayerNorm':
            # Ensure there is a preceding layer for LayerNormalization
            if len(layer_outputs) == 0:
                #raise ValueError("LayerNormalization layer must have a preceding layer.")
                config['normalization'] = None
            else:
                model.add(LayerNormalization())

        # Add residual connection if specified
        residual_type = config.get('residual', None)
        if residual_type == 'Add':
            if layer_outputs:
                if len(layer_outputs) > 1:
                    added = Add()(layer_outputs)
                else:
                    added = layer_outputs[0]
                model.add(Conv2D(filters=config.get('filters', 32), kernel_size=(1, 1), activation='relu'))  # 1x1 convolution to adjust dimensions if needed
                model.add(added)
            layer_outputs = []  # Clear layer_outputs after adding residual connection

        # Add activation layer if specified
        activation_type = config.get('activation', None)
        if activation_type:
            model.add(Activation(activation_type))

        # Store current layer output for potential residual connection
        if (len(layer_outputs) != 0):
            layer_outputs.append(model.layers[-1].output)  # Append current layer output

    # if len(layer_outputs) > 0 and len(model.layers[-1].output_shape != 2):
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    # Load dataset
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
    print(model.summary())

    model_size = get_model_memory(model, batch_size = 1)
    flops = get_flops(model, None)

    if (weights):
      return model, model.get_weights()

    _, accuracy = model.evaluate(x_test, y_test)

    return accuracy, latency, model_size, flops

def get_model_memory(model, batch_size):
  features_mem = 0
  float_bytes = 4.0
  single_layer_mem_float = 0

  for layer in model.layers:
    out_shape = layer.output.shape

    if type(out_shape) is list:   #e.g. input layer which is a list
        out_shape = out_shape[0]
    else:
        # Handle cases where output shape might not be 3D
        if len(out_shape) >= 4:
            out_shape = [out_shape[1], out_shape[2], out_shape[3]]
        else:
            out_shape = [out_shape[1]]  # Handle 1D output (e.g., after Flatten)

    single_layer_mem = 1
    for s in out_shape:
      if s is not None:
        single_layer_mem *= s
    single_layer_mem_float += single_layer_mem * float_bytes
    single_layer_mem_mb = single_layer_mem_float / (1024 * 1024)
    features_mem += single_layer_mem_mb

  trainable_wts = np.sum([K.count_params(p) for p in model.trainable_weights])
  non_trainable_wts = np.sum([K.count_params(p) for p in model.non_trainable_weights])
  parameter_mem_MB = ((trainable_wts + non_trainable_wts) * float_bytes)/(1024**2)
  total_memory_MB = (batch_size * features_mem) + parameter_mem_MB

  total_memory_GB = total_memory_MB/1024
  

  return total_memory_GB


# def get_flops(model):
#   run_meta = tf.compat.v1.RunMetadata()
#   opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#   flops = tf.compat.v1.profiler.profile(graph=model.graph, run_meta=run_meta, cmd='op', options=opts)
#   return flops.total_float_ops / 1e9  # FLOPs in billions

def get_flops(model, batch_size=None):
    if batch_size is None:
        batch_size = 1

    # Convert the tuple to a list before concatenation
    input_shape = list(model.inputs[0].shape[1:]) 
    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + input_shape, model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                            run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops
     

def selection(population, fitness_scores, num_parents):
    pareto_front_indices = pareto_front(population, fitness_scores)
    num_parents = min(num_parents, len(pareto_front_indices))  # Ensure num_parents is not larger than the Pareto front
    if len(pareto_front_indices) > num_parents:
        # If more non-dominated solutions than needed, select the top ones based on their rank
        selected_indices = sorted(pareto_front_indices, key=lambda idx: fitness_scores[idx])[:num_parents]
    else:
        selected_indices = pareto_front_indices

    return [population[idx] for idx in selected_indices]

def crossover(parents, offspring_size):
    offspring = []
    while len(offspring) < offspring_size:
        # Ensure there are enough parents for crossover
        if len(parents) >= 2: 
            parent1, parent2 = random.sample(parents, 2)

            # Ensure parents have lengths greater than 1
            if len(parent1) > 1 and len(parent2) > 1:
                crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
                offspring.append(child)
        else:
            # If not enough parents, simply copy a random parent
            offspring.append(random.choice(parents).copy()) 

    return offspring

def mutation(offspring, mutation_rate):
    for idx in range(len(offspring)):
        for arch_idx in range(len(offspring[idx])):
            layer_config = offspring[idx][arch_idx]
            if random.random() < mutation_rate:
                # Mutate layer type
                layer_config['layer_type'] = random.choice(layer_type_space)

                # Mutate other parameters based on layer type
                if layer_config['layer_type'] == 'conv' or layer_config['layer_type'] == 'depthwise_conv' or layer_config['layer_type'] == 'separable_conv':
                    layer_config['filters'] = random.choice(filters_space)
                    layer_config['kernel_size'] = random.choice(kernel_size_space)
                    layer_config['activation'] = random.choice(activation_space)

                elif layer_config['layer_type'] == 'maxpool' or layer_config['layer_type'] == 'avgpool':
                    layer_config['pool_size'] = random.choice([(s, s) for s in kernel_size_space])

                elif layer_config['layer_type'] == 'global_avgpool':
                    # No parameters to mutate for GlobalAveragePooling2D
                    pass

                elif layer_config['layer_type'] == 'identity':
                    # No parameters to mutate for identity layer
                    pass

                # Mutate normalization layer
                layer_config['normalization'] = random.choice(normalization_space)

                # Mutate residual connection
                layer_config['residual'] = random.choice(residual_space)

                # Mutate activation function
                layer_config['activation'] = random.choice(activation_space)

    return offspring

def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        architecture = []
        num_layers = NUM_LAYERS

        for _ in range(num_layers):
            layer_config = {}
            layer_config['layer_type'] = random.choice(layer_type_space)

            if layer_config['layer_type'] == 'conv' or layer_config['layer_type'] == 'depthwise_conv' or layer_config['layer_type'] == 'separable_conv':
                layer_config['filters'] = random.choice(filters_space)
                layer_config['kernel_size'] = random.choice(kernel_size_space)
                layer_config['activation'] = random.choice(activation_space)

            elif layer_config['layer_type'] == 'maxpool' or layer_config['layer_type'] == 'avgpool':
                layer_config['pool_size'] = random.choice([(s, s) for s in kernel_size_space])

            elif layer_config['layer_type'] == 'global_avgpool':
                # No additional parameters needed
                pass

            elif layer_config['layer_type'] == 'identity':
                # No additional parameters needed
                pass

            # Add normalization layer
            layer_config['normalization'] = random.choice(normalization_space)

            # Add residual connection
            layer_config['residual'] = random.choice(residual_space)

            architecture.append(layer_config)

        population.append(architecture)

    return population


def print_weights(arch):
     model, weights = evaluate_architecture(arch, weights=True, dataset=DATASET)
     return model

def pareto_dominance(s1, s2):
    try:
        dominates_s1 = all(x >= y for x, y in zip(s1, s2)) and any(x > y for x, y in zip(s1, s2))
        dominates_s2 = all(x >= y for x, y in zip(s2, s1)) and any(x > y for x, y in zip(s2, s1))
    except TypeError as e:
        print(f"Error in pareto_dominance: {e}")
        print(f"s1: {s1}, s2: {s2}")
        raise
    return dominates_s1, dominates_s2



def pareto_front(population, fitness_scores):
    pareto_front = []
    dominated_solutions = set()

    for i in range(len(fitness_scores)):
        if i in dominated_solutions:
            continue
        current_dominated = set()
        for j in range(len(fitness_scores)):
            if i != j:
                dominates_i, dominates_j = pareto_dominance(fitness_scores[i], fitness_scores[j])
                if dominates_i:
                    dominated_solutions.add(j)
                elif dominates_j:
                    current_dominated.add(j)
        if not current_dominated:
            pareto_front.append(i)

    return pareto_front

def model_to_json(model):
    json_model = {"layers": []}
    for layer in model.layers:
        layer_config = {}
        if isinstance(layer, Conv2D):
            layer_config['layer_type'] = 'conv'
            layer_config['filters'] = layer.filters
            layer_config['kernel_size'] = list(layer.kernel_size)
            layer_config['activation'] = layer.activation.__name__
        elif isinstance(layer, MaxPooling2D):
            layer_config['layer_type'] = 'maxpool'
            layer_config['pool_size'] = list(layer.pool_size)
        elif isinstance(layer, DepthwiseConv2D):
            layer_config['layer_type'] = 'depthwise_conv'
            layer_config['kernel_size'] = list(layer.kernel_size)
            layer_config['activation'] = layer.activation.__name__
        elif isinstance(layer, SeparableConv2D):
            layer_config['layer_type'] = 'separable_conv'
            layer_config['filters'] = layer.filters
            layer_config['kernel_size'] = list(layer.kernel_size)
            layer_config['activation'] = layer.activation.__name__
        elif isinstance(layer, AveragePooling2D):
            layer_config['layer_type'] = 'avgpool'
            layer_config['pool_size'] = list(layer.pool_size)
        elif isinstance(layer, GlobalAveragePooling2D):
            layer_config['layer_type'] = 'global_avgpool'
        elif isinstance(layer, BatchNormalization):
            layer_config['layer_type'] = 'batch_norm'
        elif isinstance(layer, LayerNormalization):
            layer_config['layer_type'] = 'layer_norm'
        elif isinstance(layer, Add):
            layer_config['layer_type'] = 'add'
        elif isinstance(layer, Concatenate):
            layer_config['layer_type'] = 'concatenate'
        elif isinstance(layer, Flatten):
            layer_config['layer_type'] = 'flatten'
        elif isinstance(layer, Dense):
            layer_config['layer_type'] = 'dense'
            layer_config['units'] = layer.units
            layer_config['activation'] = layer.activation.__name__
        
        json_model["layers"].append(layer_config)

    return json_model

def save_model_to_json(model, file_path):
    json_model = model_to_json(model)
    with open(file_path, 'w') as f:
        json.dump(json_model, f, indent=4)


def nas_multi_objective_genetic_algorithm(population_size, num_generations, dataset=DATASET):
    population = initialize_population(population_size)

    for generation in range(num_generations):
        # Evaluate fitness of the population
        fitness_scores = [evaluate_architecture(architecture, weights=False, dataset=dataset) for architecture in population]

        # Determine Pareto front
        pareto_front_solutions = selection(population, fitness_scores, num_parents=POPULATION_SIZE // 2)
        # Apply crossover and mutation
        parents = pareto_front_solutions
        offspring = crossover(parents, offspring_size=population_size - len(parents))
        offspring = mutation(offspring, mutation_rate=MUTATION_RATE)

        # Create new population
        population = parents + offspring

        print(f"Generation {generation + 1}, Pareto front size: {len(pareto_front_solutions)}")

    # Final Pareto front
    final_fitness_scores = [evaluate_architecture(architecture, weights=False, dataset=dataset) for architecture in population]
    final_pareto_front = pareto_front(population, final_fitness_scores)
    best_architectures = [population[idx] for idx in final_pareto_front]


    print("Best architectures based on Pareto front:")
    for arch in best_architectures:
        print(arch)
        # Optionally print weights
        print_weights(arch)



def nas_genetic_algorithm(population_size, num_generations, dataset=DATASET):
    population = initialize_population(population_size)

    for generation in range(num_generations):
        fitness_scores = [evaluate_architecture(architecture, weights=False, dataset=dataset) for architecture in population]

        parents = selection(population, fitness_scores, num_parents=10)
        offspring = crossover(parents, offspring_size=population_size - len(parents))
        offspring = mutation(offspring, mutation_rate=MUTATION_RATE)

        population = parents + offspring

        print(f"Generation {generation + 1}, Best Accuracy: {max(fitness_scores)}")

    best_architecture = population[np.argmax(fitness_scores)]
    print("Best architecture:", best_architecture)
    model = print_weights(best_architecture)
    save_model_to_json(model, 'model_config.json')

    # model.save('model.h5')
    weights = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if layer_weights:
            for i, weight in enumerate(layer_weights):
                name = layer.name + "_" + ("weight" if i == 0 else "bias")
                weights.append((name, weight))

    # Create the 'model' directory if it doesn't exist
    if not os.path.exists("./model"):
        os.makedirs("./model")

    for w in weights:
        with open("./model/" + str(w[0]) + ".npy", "wb") as f:
            print("shape")

            if "bias" in w[0]:
                weights_array = np.expand_dims(w[1], axis=(0))
                print(weights_array.shape)

                weights_array = np.ascontiguousarray(weights_array)
                np.save(f, weights_array)
            else:
                weights_array = w[1].T
                print(weights_array.shape)

                weights_array = np.ascontiguousarray(weights_array)
                np.save(f, weights_array)


# Example usage
nas_multi_objective_genetic_algorithm(population_size=POPULATION_SIZE, num_generations=NUM_GENERATIONS, dataset=DATASET)



