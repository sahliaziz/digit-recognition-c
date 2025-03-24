#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <byteswap.h>
#include "tensor.h"


Tensor *idx_to_tensor(char* filename) {
    uint16_t zero_bytes;
    uint8_t data_type;
    uint32_t to_swap;
    uint32_t size = 1;

    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Error opening file");
        fclose(f);
        return NULL;
    }

    Tensor *tensor = malloc(sizeof(Tensor));

    if (!tensor) {
        perror("Memory allocation failed");
        fclose(f);
        return NULL;
    }

    if (!fread(&zero_bytes, sizeof(uint16_t), 1, f)) {
        perror("Failed to read file");
        fclose(f);
        return NULL;
    }

    if (zero_bytes != 0) {
        perror("Invalid file format");
        free(tensor);
        fclose(f);
        return NULL;
    }

    fread(&data_type, sizeof(uint8_t), 1, f);
    fread(&tensor->n_dims, sizeof(uint8_t), 1, f);

    // Allocate memory for dimension sizes
    tensor->shape = malloc(tensor->n_dims * sizeof(uint32_t));
    if (!tensor->shape) {
        perror("Memory allocation failed");
        free(tensor);
        fclose(f);
        return NULL;
    }

    // Reading dimensions
    for (uint32_t i = 0; i < tensor->n_dims; i++) {
        fread(&to_swap, sizeof(uint32_t), 1, f);
        tensor->shape[i] = __bswap_32(to_swap);
        size *= tensor->shape[i];
    }

    // First dimension represents the number of images
    uint32_t n_examples = tensor->shape[0];
    tensor->size = size;


    // Allocating memory for image data (28x28 pixels per image)
    tensor->data = malloc(size * sizeof(double));
    if (!tensor->data) {
        perror("Memory allocation failed");
        free(tensor->shape);
        free(tensor);
        fclose(f);
        return NULL;
    }

    uint8_t *buf = malloc(size * sizeof(uint8_t));

    // Reading all image data to the buffer first
    fread(buf, 1, size, f);

    // Transfering image data to tensor as double
    for (uint32_t i = 0; i<size; i++) {
        tensor->data[i] = buf[i] / 255.0;
    }

    free(buf);
    fclose(f);

    return tensor;
}

Tensor *copy_tensor(Tensor *tensor) {
    Tensor *result = malloc(sizeof(Tensor));
    if (!result) {
        perror("Memory allocation failed");
        return NULL;
    }

    result->size = tensor->size;
    result->n_dims = tensor->n_dims;
    result->shape = malloc(result->n_dims * sizeof(uint32_t));
    result->data = malloc(result->size * sizeof(double));
    
    if (!result->shape) {
        perror("Memory allocation failed");
        free(result);
        return NULL;
    }

    for (uint32_t i = 0; i < tensor->n_dims; i++) {
        result->shape[i] = tensor->shape[i];
    }
    
    for (uint32_t i = 0; i < tensor->size; i++) {
        result->data[i] = tensor->data[i];
    }

    return result;
}

Tensor *scale(Tensor *tensor, double factor) {
    Tensor *result = copy_tensor(tensor);

    for (uint32_t i = 0; i < tensor->size; i++) {
        result->data[i] = tensor->data[i] * factor;
    }

    return result;
}

Tensor *matmul(Tensor *matrix1, Tensor *matrix2) {
    if (matrix1->n_dims != 2 || matrix2->n_dims != 2) {
        perror("Cannot multiply tensors of dimensions different than 2");
        return NULL;
    }

    if (matrix1->shape[1] != matrix2->shape[0]) {
        perror("Tensor shapes are not compatible");
        return NULL;
    }

    Tensor *result = malloc(sizeof(Tensor));
    if (!result) {
        perror("Memory allocation failed");
        return NULL;
    }

    result->size = matrix1->shape[0] * matrix2->shape[1];
    result->n_dims = 2;
    result->shape = malloc(2 * sizeof(uint32_t));
    if (!result->shape) {
        perror("Memory allocation failed");
        free(result);
        return NULL;
    }

    result->shape[0] = matrix1->shape[0];
    result->shape[1] = matrix2->shape[1];
    result->size = result->shape[0] * result->shape[1];

    result->data = malloc(result->size * sizeof(double));
    if (!result->data) {
        perror("Memory allocation failed");
        free(result->shape);
        free(result);
        return NULL;
    }

    for (uint32_t i = 0; i < result->shape[0]; i++) {
        for (uint32_t j = 0; j < result->shape[1]; j++) {
            result->data[i * result->shape[1] + j] = 0;
            for (uint32_t k = 0; k < matrix1->shape[1]; k++) {
                result->data[i * result->shape[1] + j] += matrix1->data[i * matrix1->shape[1] + k] * matrix2->data[k * matrix2->shape[1] + j];
            }
        }
    }

    return result;
}

Tensor *transpose(Tensor *tensor) {
    if (tensor->n_dims != 2) {
        perror("Cannot transpose tensor of dimensions different than 2");
        return NULL;
    }

    Tensor *result = malloc(sizeof(Tensor));

    if (!result) {
        perror("Memory allocation failed");
        return NULL;
    }
    
    result->size = tensor->size;
    result->n_dims = 2;
    result->shape = malloc(2 * sizeof(uint32_t));

    if (!result->shape) {
        perror("Memory allocation failed");
        free(result);
        return NULL;
    }

    result->shape[0] = tensor->shape[1];
    result->shape[1] = tensor->shape[0];
    result->size = result->shape[0] * result->shape[1];

    result->data = malloc(result->size * sizeof(double));
    if (!result->data) {
        perror("Memory allocation failed");
        free(result->shape);
        free(result);
        return NULL;
    }

    for (uint32_t i = 0; i < tensor->shape[0]; i++) {
        for (uint32_t j = 0; j < tensor->shape[1]; j++) {
            result->data[j * result->shape[1] + i] = tensor->data[i * tensor->shape[1] + j];
        }
    }
    
    return result;
}

void free_tensor(Tensor *tensor) {
    free(tensor->shape);
    free(tensor->data);
    free(tensor);
    tensor = NULL;
}

double sum(Tensor *tensor) {
    double sum = 0;
    for (uint32_t i = 0; i < tensor->size; i++) {
        sum += tensor->data[i];
    }
    return sum;
}

void reshape_tensor(Tensor *tensor, uint32_t n, uint32_t m) {
    if (n * m != tensor->size) {
        perror("Could not reshape tensor");
        return;
    }

    tensor->n_dims = 2;
    free(tensor->shape); // Free the old shape memory
    tensor->shape = malloc(2 * sizeof(uint32_t));
    if (!tensor->shape) {
        perror("Memory allocation failed");
        return;
    }
    
    tensor->shape[0] = n;
    tensor->shape[1] = m;
    tensor->size = n * m;

    return;
}

Tensor *argmax(Tensor *tensor) {
    Tensor *result = malloc(sizeof(Tensor));
    if (!result) {
        perror("Memory allocation failed");
        return NULL;
    }

    result->size = tensor->shape[0];
    result->n_dims = 2;
    result->shape = malloc(2 * sizeof(uint32_t));
    if (!result->shape) {
        perror("Memory allocation failed");
        free(result);
        return NULL;
    }

    result->shape[0] = tensor->shape[0];
    result->shape[1] = 1;

    result->data = malloc(result->size * sizeof(double));
    if (!result->data) {
        perror("Memory allocation failed");
        free(result->shape);
        free(result);
        return NULL;
    }

    for (uint32_t i = 0; i < tensor->shape[0]; i++) {
        double max = tensor->data[i * tensor->shape[1]];
        uint32_t max_index = 0;
        for (uint32_t j = 1; j < tensor->shape[1]; j++) {
            if (tensor->data[i * tensor->shape[1] + j] > max) {
                max = tensor->data[i * tensor->shape[1] + j];
                max_index = j;
            }
        }
        result->data[i] = (double)max_index;
    }

    return result;
}

Tensor *random_tensor(uint32_t n, uint32_t m) {
    Tensor *tensor = malloc(sizeof(Tensor));
    if (!tensor) {
        perror("Memory allocation failed");
        return NULL;
    }

    tensor->size = n * m;
    tensor->n_dims = 2;
    tensor->shape = malloc(2 * sizeof(uint32_t));
    if (!tensor->shape) {
        perror("Memory allocation failed");
        free(tensor);
        return NULL;
    }

    tensor->shape[0] = n;
    tensor->shape[1] = m;

    tensor->data = malloc(tensor->size * sizeof(double));
    if (!tensor->data) {
        perror("Memory allocation failed");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    for (uint32_t i = 0; i < tensor->size; i++) {
        tensor->data[i] = (double)rand() / RAND_MAX;
    }

    return tensor;
}

Tensor *ReLU(Tensor *tensor) {
    Tensor *result = malloc(sizeof(Tensor));
    if (!result) {
        perror("Memory allocation failed");
        return NULL;
    }

    result->size = tensor->size;
    result->n_dims = tensor->n_dims;
    result->shape = malloc(tensor->n_dims * sizeof(uint32_t));
    if (!result->shape) {
        perror("Memory allocation failed");
        free(result);
        return NULL;
    }

    for (uint32_t i = 0; i < tensor->n_dims; i++) {
        result->shape[i] = tensor->shape[i];
    }

    result->data = malloc(result->size * sizeof(double));
    if (!result->data) {
        perror("Memory allocation failed");
        free(result->shape);
        free(result);
        return NULL;
    }

    for (uint32_t i = 0; i < tensor->size; i++) {
        result->data[i] = tensor->data[i] > 0 ? tensor->data[i] : 0;
    }

    return result;
}

Tensor *softmax(Tensor *tensor) {
    Tensor *result = copy_tensor(tensor);

    for (uint32_t i = 0; i < tensor->shape[0]; i++) {
        double max = tensor->data[i * tensor->shape[1]];
        for (uint32_t j = 1; j < tensor->shape[1]; j++) {
            if (tensor->data[i * tensor->shape[1] + j] > max) {
                max = tensor->data[i * tensor->shape[1] + j];
            }
        }

        double sum = 0;
        for (uint32_t j = 0; j < tensor->shape[1]; j++) {
            result->data[i * tensor->shape[1] + j] = exp(tensor->data[i * tensor->shape[1] + j] - max);
            sum += result->data[i * tensor->shape[1] + j];
        }

        for (uint32_t j = 0; j < tensor->shape[1]; j++) {
            result->data[i * tensor->shape[1] + j] /= sum;
        }
    }

    return result;
}

Tensor *add(Tensor *tensor1, Tensor *tensor2) {
    if (tensor1->size != tensor2->size) {
        perror("Cannot add tensors of different sizes");
        return NULL;
    }

    Tensor *result = malloc(sizeof(Tensor));
    if (!result) {
        perror("Memory allocation failed");
        return NULL;
    }

    result->size = tensor1->size;
    result->n_dims = tensor1->n_dims;
    result->shape = malloc(tensor1->n_dims * sizeof(uint32_t));
    if (!result->shape) {
        perror("Memory allocation failed");
        free(result);
        return NULL;
    }

    for (uint32_t i = 0; i < tensor1->n_dims; i++) {
        result->shape[i] = tensor1->shape[i];
    }

    result->data = malloc(result->size * sizeof(double));
    if (!result->data) {
        perror("Memory allocation failed");
        free(result->shape);
        free(result);
        return NULL;
    }

    for (uint32_t i = 0; i < tensor1->size; i++) {
        result->data[i] = tensor1->data[i] + tensor2->data[i];
    }

    return result;
}

Tensor *matrix_add_bias(Tensor *matrix, Tensor *bias) {
    if (matrix->n_dims != 2 || bias->shape[1] != 1 || matrix->shape[1] != bias->shape[0]) {
        perror("Shapes are not compatible for bias addition");
        return NULL;
    }

    Tensor *result = malloc(sizeof(Tensor));
    if (!result) {
        perror("Memory allocation failed");
        return NULL;
    }

    result->size = matrix->size;
    result->n_dims = 2;
    result->shape = malloc(2 * sizeof(uint32_t));
    if (!result->shape) {
        perror("Memory allocation failed");
        free(result);
        return NULL;
    }

    result->shape[0] = matrix->shape[0];
    result->shape[1] = matrix->shape[1];

    result->data = malloc(result->size * sizeof(double));
    if (!result->data) {
        perror("Memory allocation failed");
        free(result->shape);
        free(result);
        return NULL;
    }

    for (uint32_t i = 0; i < matrix->shape[0]; i++) {
        for (uint32_t j = 0; j < matrix->shape[1]; j++) {
            result->data[i * matrix->shape[1] + j] = matrix->data[i * matrix->shape[1] + j] + bias->data[j];
        }
    }

    return result;
}

Tensor *one_hot(Tensor *labels) {
    Tensor *result = malloc(sizeof(Tensor));
    if (!result) {
        perror("Memory allocation failed");
        return NULL;
    }

    result->size = 10 * labels->size;
    result->n_dims = 2;
    result->shape = malloc(sizeof(uint32_t));
    if (!result->shape) {
        perror("Memory allocation failed");
        free(result);
        return NULL;
    }

    result->shape[0] = labels->size;
    result->shape[1] = 10;

    result->data = malloc(result->size * sizeof(double));
    if (!result->data) {
        perror("Memory allocation failed");
        free(result->shape);
        free(result);
        return NULL;
    }

    for (uint32_t i = 0; i < labels->size; i++) {
        for (uint32_t j = 0; j < 10; j++) {
            result->data[i * 10 + j] = j == labels->data[i] ? 1.0 : 0.0;
        }
    }

    return result;
}

int main() {

    Tensor *X_test = idx_to_tensor("train-images.idx3-ubyte");
    Tensor *Y_test = idx_to_tensor("train-labels.idx1-ubyte");

    reshape_tensor(X_test, 60000, 784);

    Tensor *W1 = random_tensor(784, 128);
    Tensor *b1 = random_tensor(128, 1);
    Tensor *W2 = random_tensor(128, 10);
    Tensor *b2 = random_tensor(10, 1);

    Tensor *Y_enc = one_hot(Y_test);

    Tensor *Z1 = matrix_add_bias(matmul(X_test, W1), b1);
    Tensor *A1 = ReLU(Z1);

    Tensor *Z2 = matrix_add_bias(matmul(A1, W2), b2);

    Tensor *A2 = softmax(Z2);

    Tensor *Y_pred = argmax(A2);

    Tensor *d_pred = add(Y_enc, scale(Y_pred, -1));
    Tensor *dW2 = matmul(transpose(A1), d_pred);
    
}