import random
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, DepthwiseConv2D, SeparableConv2D, GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, LayerNormalization, Activation, Concatenate, Add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


NUM_LAYERS = 3

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
def evaluate_architecture(architecture, dataset='mnist'):
    model = Sequential()
    # inputs = Input(shape=(28, 28, 1))  # Assuming MNIST image shape
    layer_outputs = []  # To store outputs for potential concatenation or residual connection
    
    for config in architecture:
        layer_type = config['layer_type']
        
        if layer_type == 'conv':

            model.add(Conv2D(filters=config.get('filters', 32), kernel_size=config.get('kernel_size', (3, 3)), activation=config.get('activation', 'relu'), input_shape=config.get('input_shape', None)))
        elif layer_type == 'maxpool':
            # Ensure input dimensions are sufficient for pooling
            if len(layer_outputs) != 0 and model.layers[-1].output_shape[1] >= 5 and model.layers[-1].output_shape[2] >= 5:
                model.add(MaxPooling2D(pool_size=config.get('pool_size', (2, 2))))
            else:
                # Skip pooling if input dimensions are too small
                pass
        elif layer_type == 'depthwise_conv':
            model.add(DepthwiseConv2D(kernel_size=config.get('kernel_size', (3, 3)), activation=config.get('activation', 'relu')))
        elif layer_type == 'separable_conv':
            model.add(SeparableConv2D(filters=config.get('filters', 32), kernel_size=config.get('kernel_size', (3, 3)), activation=config.get('activation', 'relu')))
        elif layer_type == 'avgpool':
            if len(layer_outputs) != 0 and model.layers[-1].output_shape[1] >= 5 and model.layers[-1].output_shape[2] >= 5:
                model.add(AveragePooling2D(pool_size=config.get('pool_size', (2, 2))))
            else:
                # Skip pooling if input dimensions are too small
                pass
        elif layer_type == 'global_avgpool':
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
                model.add(added)
            layer_outputs = []  # Clear layer_outputs after adding residual connection
        
        # Add activation layer if specified
        activation_type = config.get('activation', None)
        if activation_type:
            model.add(Activation(activation_type))
        
        # Store current layer output for potential residual connection
        if (len(layer_outputs) != 0):
            layer_outputs.append(model.layers[-1].output)  # Append current layer output
    
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))  # 10 classes for both MNIST and CIFAR-10
    
    # Load dataset
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train / 255.0, axis=-1)
        x_test = np.expand_dims(x_test / 255.0, axis=-1)
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0
    
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping], verbose=0)
    print(model.summary())
    _, accuracy = model.evaluate(x_test, y_test)
    
    return accuracy



# Genetic algorithm functions
def selection(population, fitness_scores, num_parents):
    selected_indices = np.argsort(fitness_scores)[-num_parents:]
    return [population[idx] for idx in selected_indices]

def crossover(parents, offspring_size):
    offspring = []
    while len(offspring) < offspring_size:
        parent1, parent2 = random.sample(parents, 2)
        
        # Ensure parents have lengths greater than 1
        if len(parent1) > 1 and len(parent2) > 1:
            crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            offspring.append(child)
    
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



# Example of NAS using genetic algorithm
def nas_genetic_algorithm(population_size, num_generations, dataset='mnist'):
    population = initialize_population(population_size)
    
    for generation in range(num_generations):
        fitness_scores = [evaluate_architecture(architecture, dataset=dataset) for architecture in population]
        
        parents = selection(population, fitness_scores, num_parents=10)
        offspring = crossover(parents, offspring_size=population_size - len(parents))
        offspring = mutation(offspring, mutation_rate=0.1)
        
        population = parents + offspring
        
        print(f"Generation {generation + 1}, Best Accuracy: {max(fitness_scores)}")
    
    best_architecture = population[np.argmax(fitness_scores)]
    print("Best architecture:", best_architecture)

# Example usage
nas_genetic_algorithm(population_size=3, num_generations=2, dataset='mnist')
