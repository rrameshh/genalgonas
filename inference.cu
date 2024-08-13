#include <jansson.h> // Library for JSON parsing
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cuda_runtime.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// Load JSON configuration
json_t* load_json(const char* file_path) {
    json_error_t error;
    json_t* json = json_load_file(file_path, 0, &error);
    if (!json) {
        fprintf(stderr, "Error loading JSON file: %s\n", error.text);
        return NULL;
    }
    return json;
}

// Load weights
float* load_weights(const char* file_path, int* rows, int* cols) {
    PyArrayObject* array = read_numpy_file(file_path);
    if (array == NULL) return NULL;

    int ndims = PyArray_NDIM(array);
    npy_intp* dims = PyArray_DIMS(array);

    if (ndims != 2) {
        fprintf(stderr, "Expected 2D array.\n");
        Py_DECREF(array);
        return NULL;
    }

    if (rows) *rows = (int)dims[0];
    if (cols) *cols = (int)dims[1];

    float* weights = convert_PyArrayObject_to_float(array, NULL, NULL);
    Py_DECREF(array);

    return weights;
}

// Parse model configuration and initialize model structure
Model loadModel(const char* config_path) {
    Model model;
    json_t* config = load_json(config_path);
    if (!config) {
        model.layer_count = 0;
        model.layers = NULL;
        return model;
    }

    json_t* layers = json_object_get(config, "layers");
    model.layer_count = json_array_size(layers);
    model.layers = (Layer*)malloc(model.layer_count * sizeof(Layer));

    for (int i = 0; i < model.layer_count; ++i) {
        json_t* layer = json_array_get(layers, i);
        const char* type = json_string_value(json_object_get(layer, "type"));
        model.layers[i].type = strdup(type);
        model.layers[i].shape_size = json_array_size(json_object_get(layer, "shape"));

        model.layers[i].shape = (int*)malloc(model.layers[i].shape_size * sizeof(int));
        json_t* shape = json_object_get(layer, "shape");
        for (int j = 0; j < model.layers[i].shape_size; ++j) {
            model.layers[i].shape[j] = (int)json_integer_value(json_array_get(shape, j));
        }

        // Load weights and biases
        if (strcmp(type, "dense") == 0) {
            model.layers[i].weights = load_weights(json_string_value(json_object_get(layer, "weights")), NULL, NULL);
            model.layers[i].biases = load_weights(json_string_value(json_object_get(layer, "biases")), NULL, NULL);
        }
    }

    json_decref(config);
    return model;
}

// Structure for model layers
typedef struct {
    char* type; // e.g., "dense", "relu", "softmax"
    int* shape; // dimensions of the layer
    int shape_size; // number of dimensions
    float* weights; // pointer to weights if applicable
    float* biases; // pointer to biases if applicable
} Layer;

// Structure for the model
typedef struct {
    Layer* layers; // list of layers
    int layer_count; // number of layers
} Model;

// Function to read NumPy file and return as PyArrayObject
PyArrayObject* read_numpy_file(const char* file_path) {
    PyObject* numpy_module = PyImport_ImportModule("numpy");
    if (numpy_module == NULL) {
        PyErr_Print();
        return NULL;
    }
    
    PyObject* numpy_function = PyObject_GetAttrString(numpy_module, "load");
    if (numpy_function == NULL) {
        PyErr_Print();
        Py_DECREF(numpy_module);
        return NULL;
    }
    
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(file_path));
    
    PyObject* result = PyObject_CallObject(numpy_function, args);
    if (result == NULL) {
        PyErr_Print();
        Py_DECREF(numpy_module);
        Py_DECREF(numpy_function);
        Py_DECREF(args);
        return NULL;
    }
    
    Py_DECREF(numpy_module);
    Py_DECREF(numpy_function);
    Py_DECREF(args);
    
    return (PyArrayObject*)result;
}

// Function to convert PyArrayObject to float array
float* convert_PyArrayObject_to_float(PyArrayObject* array, int* shape, int* ndim) {
    if (PyArray_TYPE(array) != NPY_FLOAT32) {
        fprintf(stderr, "Input numpy array is not of type float.\n");
    }
    
    if (ndim) {
        *ndim = PyArray_NDIM(array);
    }
    
    return (float*)PyArray_DATA(array);
}

// Function to load weights from NumPy files and return as float array
float* load_weights(const char* file_path, int* rows, int* cols) {
    PyArrayObject* array = read_numpy_file(file_path);
    if (array == NULL) {
        return NULL;
    }
    
    int ndims = PyArray_NDIM(array);
    npy_intp* dims = PyArray_DIMS(array);
    
    if (ndims != 2) {
        fprintf(stderr, "Expected 2D array.\n");
        Py_DECREF(array);
        return NULL;
    }
    
    if (rows) *rows = (int)dims[0];
    if (cols) *cols = (int)dims[1];
    
    float* weights = convert_PyArrayObject_to_float(array, NULL, NULL);
    Py_DECREF(array);
    
    return weights;
}

__global__ void depthwiseConvKernel(float* input, float* weights, float* output, int inputChannels, int outputChannels, int kernelSize, int inputWidth, int inputHeight, string activation) {
    // Calculate output index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < inputWidth && y < inputHeight && c < outputChannels) {
        float sum = 0.0f;
        int kernelOffset = kernelSize / 2;

        // Perform convolution for each channel separately
        for (int ky = -kernelOffset; ky <= kernelOffset; ++ky) {
            for (int kx = -kernelOffset; kx <= kernelOffset; ++kx) {
                int inputX = x + kx;
                int inputY = y + ky;

                if (inputX >= 0 && inputX < inputWidth && inputY >= 0 && inputY < inputHeight) {
                    int inputIndex = (inputY * inputWidth + inputX) * inputChannels + c;
                    int weightIndex = ((ky + kernelOffset) * kernelSize + (kx + kernelOffset)) * inputChannels + c;
                    sum += input[inputIndex] * weights[weightIndex];
                }
            }
        }

        int outputIndex = (y * inputWidth + x) * outputChannels + c;
        activation(output, outputIndex, sum, activation); 
            
    }
}

__device__ void activation(float* output, int outputIndex, float sum, string activation)
{
    if (strcmp(activation, 'relu6') == 0)
        output[outputIndex] = fminf(fmaxf(sum, 0.0f), 6.0f);
    else if (strcmp(activation, 'relu') == 0)
        output[outputIndex] = fmaxf(0.0f, sum);
    else if (strcmp(activation, 'silu') == 0)
        output[outputIndex] =  sum / (1.0 + exp(sum));
    else if (strcmp(activation, 'sigmoid') == 0)
        output[outputIndex] = 1.0 / (1.0 + exp(-sum));
    else if (strcmp(activation, 'swish') == 0)
        
    
    // else if (strcmp(activ))
}


__global__ void maxpooling_2d(float *input, float *output, int N, int stride ) {
  // Calculate the global thread positions
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int start_r = row * stride - MASK_OFFSET;
  int start_c = col * stride - MASK_OFFSET;


  float maxval = FLT_MIN;

  // Iterate over all the rows
  for (int i = 0; i < MASK_DIM; i++) {
    // Go over each column
    for (int j = 0; j < MASK_DIM; j++) {
      // Range check for rows
      if ((start_r + i) >= 0 && (start_r + i) < N) {
        // Range check for columns
        if ((start_c + j) >= 0 && (start_c + j) < N) {
          if (input[(start_r + i) * N + (start_c + j)] > maxval) {
            maxval = input[(start_r + i) * N + (start_c + j)];
          }
        }
      }
    }
  }

  output[row * N + col] = maxval;

}

__global__ void avgpooling_2d(float *input, float *output, int N, int stride ) {
  // Calculate the global thread positions
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  printf("%d", row);
  printf("%d", col);

  int start_r = row * stride - MASK_OFFSET;
  int start_c = col * stride - MASK_OFFSET;


  float total = 0;

  // Iterate over all the rows
  for (int i = 0; i < MASK_DIM; i++) {
    // Go over each column
    for (int j = 0; j < MASK_DIM; j++) {
      // Range check for rows
      if ((start_r + i) >= 0 && (start_r + i) < N) {
        // Range check for columns
        if ((start_c + j) >= 0 && (start_c + j) < N) {
          total += input[(start_r + i) * N + (start_c + j)];
        }
      }
    }
  }

  output[row * N + col] = total/(MASK_DIM*MASK_DIM);

}

__global__ void separable_conv2d(float *output, float *result, float *depthwise_filter, float *pointwise_filter, int N, int in_channels, int out_channels, int filter_size) {
  // Calculate the global thread positions
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int channel = blockIdx.z * blockDim.z + threadIdx.z;

  // Depthwise Convolution
  // Starting index for calculation
  int start_r = row - filter_size / 2;
  int start_c = col - filter_size / 2;

  // Temp value for accumulating the result
  float temp_depthwise = 0;

  // Iterate over the filter
  for (int i = 0; i < filter_size; i++) {
    for (int j = 0; j < filter_size; j++) {
      // Range check for rows and columns
      if ((start_r + i) >= 0 && (start_r + i) < N && (start_c + j) >= 0 && (start_c + j) < N) {
        // Accumulate result
        temp_depthwise += input[(start_r + i) * N * in_channels + (start_c + j) * in_channels + channel] *
                depthwise_filter[i * filter_size + j];
      }
    }
  }

  // Intermediate result for depthwise convolution
  float depthwise_output = temp_depthwise;

  // Pointwise Convolution
  // Temp value for accumulating the result
  float temp_pointwise = 0;

  // Iterate over the input channels
  for (int in_channel = 0; in_channel < in_channels; in_channel++) {
    // Accumulate result
    temp_pointwise += depthwise_output *
            pointwise_filter[in_channel * out_channels + channel];
  }

  // Write back the final result
  output[row * N * out_channels + col * out_channels + channel] = temp_pointwise;
}

__global__ void batchnorm(float *input, float *output, float *mean, float *variance, float *gamma, float *beta, int N, int channels, float epsilon) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int channel = index % channels;
  int batch_index = index / channels;

  // Normalize the input
  output[index] = (input[index] - mean[channel]) / sqrt(variance[channel] + epsilon);

  // Scale and shift using gamma and beta
  output[index] = output[index] * gamma[channel] + beta[channel];
}

__global__ void layernorm(float *input, float *output, float *gamma, float *beta, int N, int C, float epsilon) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int c = index % C;
    int n = index / C;

    // Calculate mean and variance for each sample (across channels)
    float sum = 0.0;
    float sq_sum = 0.0;
    for (int i = 0; i < C; i++) {
        int idx = n * C + i;
        sum += input[idx];
        sq_sum += input[idx] * input[idx];
    }
    float mean = sum / C;
    float variance = (sq_sum / C) - (mean * mean);

    // Normalize, scale and shift
    output[index] = (input[index] - mean) / sqrt(variance + epsilon);
    output[index] = output[index] * gamma[c] + beta[c];
}

__global__ void conv(float *input, float *output, float* weights, float* biases, int N)
{
    // Calculate the global thread positions
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Starting index for calculation
  int start_r = row - MASK_OFFSET;
  int start_c = col - MASK_OFFSET;

  // Temp value for accumulating the result
  int temp = 0;

  // Iterate over all the rows
  for (int i = 0; i < MASK_DIM; i++) {
    // Go over each column
    for (int j = 0; j < MASK_DIM; j++) {
      // Range check for rows
      if ((start_r + i) >= 0 && (start_r + i) < N) {
        // Range check for columns
        if ((start_c + j) >= 0 && (start_c + j) < N) {
          // Accumulate result
          temp += input[(start_r + i) * N + (start_c + j)] *
                  weights[i * MASK_DIM + j];
        }
      }
    }
  }

  // Write back the result
  output[row * N + col] = temp;
}




// CUDA kernel for dense layer
__global__ void denseLayerKernel(float* input, float* weights, float* biases, float* output, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[i * outputSize + idx];
        }
        output[idx] = sum + biases[idx];
    }
}

// Function to call maxpooling_2d kernel
void maxpooling_2d_cpu(float* input, float* output, int N, int stride) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * N * sizeof(float));
    cudaMalloc(&d_output, N * N * sizeof(float));
    
    cudaMemcpy(d_input, input, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    maxpooling_2d<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N, stride);
    
    cudaMemcpy(output, d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

// Function to call avgpooling_2d kernel
void avgpooling_2d_cpu(float* input, float* output, int N, int stride) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * N * sizeof(float));
    cudaMalloc(&d_output, N * N * sizeof(float));
    
    cudaMemcpy(d_input, input, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    avgpooling_2d<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N, stride);
    
    cudaMemcpy(output, d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

// Function to call separable_conv2d kernel
void separable_conv2d_cpu(float* output, float* result, float* depthwise_filter, float* pointwise_filter, int N, int in_channels, int out_channels, int filter_size) {
    float *d_input, *d_output, *d_depthwise_filter, *d_pointwise_filter;
    
    cudaMalloc(&d_input, N * N * in_channels * sizeof(float));
    cudaMalloc(&d_output, N * N * out_channels * sizeof(float));
    cudaMalloc(&d_depthwise_filter, filter_size * filter_size * in_channels * sizeof(float));
    cudaMalloc(&d_pointwise_filter, in_channels * out_channels * sizeof(float));
    
    cudaMemcpy(d_input, result, N * N * in_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_depthwise_filter, depthwise_filter, filter_size * filter_size * in_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointwise_filter, pointwise_filter, in_channels * out_channels * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y, out_channels);
    
    separable_conv2d<<<numBlocks, threadsPerBlock>>>(d_output, d_input, d_depthwise_filter, d_pointwise_filter, N, in_channels, out_channels, filter_size);
    
    cudaMemcpy(output, d_output, N * N * out_channels * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_depthwise_filter);
    cudaFree(d_pointwise_filter);
}

// Function to call batchnorm kernel
void batchnorm_cpu(float* input, float* output, float* mean, float* variance, float* gamma, float* beta, int N, int channels, float epsilon) {
    float *d_input, *d_output, *d_mean, *d_variance, *d_gamma, *d_beta;
    
    cudaMalloc(&d_input, N * channels * sizeof(float));
    cudaMalloc(&d_output, N * channels * sizeof(float));
    cudaMalloc(&d_mean, channels * sizeof(float));
    cudaMalloc(&d_variance, channels * sizeof(float));
    cudaMalloc(&d_gamma, channels * sizeof(float));
    cudaMalloc(&d_beta, channels * sizeof(float));
    
    cudaMemcpy(d_input, input, N * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, mean, channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variance, variance, channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, channels * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * channels + threadsPerBlock - 1) / threadsPerBlock;
    
    batchnorm<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_mean, d_variance, d_gamma, d_beta, N, channels, epsilon);
    
    cudaMemcpy(output, d_output, N * channels * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_variance);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}

// Function to call layernorm kernel
void layernorm_cpu(float* input, float* output, float* gamma, float* beta, int N, int C, float epsilon) {
    float *d_input, *d_output, *d_gamma, *d_beta;
    
    cudaMalloc(&d_input, N * C * sizeof(float));
    cudaMalloc(&d_output, N * C * sizeof(float));
    cudaMalloc(&d_gamma, C * sizeof(float));
    cudaMalloc(&d_beta, C * sizeof(float));
    
    cudaMemcpy(d_input, input, N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, C * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * C + threadsPerBlock - 1) / threadsPerBlock;
    
    layernorm<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_gamma, d_beta, N, C, epsilon);
    
    cudaMemcpy(output, d_output, N * C * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}

// Function to call conv kernel

void conv_cpu(float* input, float* output, float* weights, float* biases, int N) {
    float *d_input, *d_output, *d_weights, *d_biases;

    // Allocate device memory
    cudaMalloc(&d_input, N * N * sizeof(float));
    cudaMalloc(&d_output, N * N * sizeof(float));
    cudaMalloc(&d_weights, N * N * sizeof(float)); // Assuming weights are N x N for simplicity
    cudaMalloc(&d_biases, N * sizeof(float)); // Assuming biases are 1D array of size N

    // Copy data from host to device
    cudaMemcpy(d_input, input, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the convolution kernel (placeholder, adjust grid and block sizes as needed)
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    conv_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_weights, d_biases, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Copy result from device to host
    cudaMemcpy(output, d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
}
// Function to execute dense layer
void denseLayer(float* input, float* weights, float* biases, float* output, int inputSize, int outputSize) {
    float* d_input;
    float* d_weights;
    float* d_biases;
    float* d_output;
    
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_weights, inputSize * outputSize * sizeof(float));
    cudaMalloc(&d_biases, outputSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));
    
    cudaMemcpy(d_input, input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, inputSize * outputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, outputSize * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (outputSize + threadsPerBlock - 1) / threadsPerBlock;
    denseLayerKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_weights, d_biases, d_output, inputSize, outputSize);
    
    cudaMemcpy(output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_output);
}

// Function to run the model
void runModel(const Model* model, float* input, float* output) {
    float* layerInput = input;
    float* layerOutput = NULL;
    
    for (int i = 0; i < model->layer_count; ++i) {
        Layer* layer = &model->layers[i];
        
        if (strcmp(layer->type, "dense") == 0) {
            layerOutput = (float*)malloc(layer->shape[1] * sizeof(float));
            denseLayer(layerInput, layer->weights, layer->biases, layerOutput, layer->shape[0], layer->shape[1]);
            layerInput = layerOutput;
        }
        else if (strcmp(layer->type, "maxpool") == 0) {
            layerOutput = (float*)malloc(layer->shape[1] * sizeof(float));
            maxpooling_2d_cpu(layerInput, layer->weights, layer->biases, layerOutput, layer->shape[0], layer->shape[1]);
            layerInput = layerOutput;
        }
        else if (strcmp(layer->type, "depthwise_conv") == 0) {
            layerOutput = (float*)malloc(layer->shape[1] * sizeof(float));
            depthwise_2d_cpu(layerInput, layer->weights, layer->biases, layerOutput, layer->shape[0], layer->shape[1]);
            layerInput = layerOutput;
        }
         else if (strcmp(layer->type, "separable_conv") == 0) {
            layerOutput = (float*)malloc(layer->shape[1] * sizeof(float));
            separable_cpu(layerInput, layer->weights, layer->biases, layerOutput, layer->shape[0], layer->shape[1]);
            layerInput = layerOutput;
        }
        else if (strcmp(layer->type, "avgpool") == 0) {
            layerOutput = (float*)malloc(layer->shape[1] * sizeof(float));
            avgpool_cpu(layerInput, layer->weights, layer->biases, layerOutput, layer->shape[0], layer->shape[1]);
            layerInput = layerOutput;
        }
        else if (strcmp(layer->type, "global_avgpool") == 0) {
            layerOutput = (float*)malloc(layer->shape[1] * sizeof(float));
            global_avgpool_cpu(layerInput, layer->weights, layer->biases, layerOutput, layer->shape[0], layer->shape[1]);
            layerInput = layerOutput;
        }    
        // Add more layer types as needed
    }
    
    memcpy(output, layerInput, model->layers[model->layer_count - 1].shape[1] * sizeof(float));
    free(layerOutput);
}

int main() {
    Py_Initialize();
    import_array();
    
    Model model = loadModel();
    
    // Load input image
    int inputSize = 784;
    float* input = (float*)malloc(inputSize * sizeof(float));
    
    // Initialize input with dummy data (replace with actual data loading)
    for (int i = 0; i < inputSize; ++i) {
        input[i] = 1.0f; // Example value
    }
    
    // Allocate memory for output
    int outputSize = model.layers[model.layer_count - 1].shape[1];
    float* output = (float*)malloc(outputSize * sizeof(float));
    
    // Run model
    runModel(&model, input, output);
    
    // Print output
    for (int i = 0; i < outputSize; ++i) {
        printf("%f ", output[i]);
    }
    printf("\n");
    
    // Clean up
    free(input);
    free(output);
    
    for (int i = 0; i < model.layer_count; ++i) {
        free(model.layers[i].shape);
        free(model.layers[i].weights);
        free(model.layers[i].biases);
    }
    free(model.layers);
    
    Py_Finalize();
    return 0;
}