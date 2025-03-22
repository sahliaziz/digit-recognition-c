#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <byteswap.h>
#include "tensor.h"


Tensor *idx_to_tensor(char* filename) {
    uint16_t zero_bytes;
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

    fread(&tensor->data_type, sizeof(uint8_t), 1, f);
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

    result->data_type = 0;
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
                uint32_t l = i * result->shape[1];
                result->data[l + j] += matrix1->data[l + k] * matrix2->data[k * matrix2->shape[1] + j];
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
    
    result->data_type = 0;
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

int main() {
    Tensor *X_test = idx_to_tensor("t10k-images.idx3-ubyte");

    // Printing some information for debugging
    printf("Data type: %d\n", X_test->data_type);
    printf("Number of dimensions: %d\n", X_test->n_dims);
    printf("Number of pixels: %d\n", X_test->size);

    printf("Dimensions: ");
    for (int i = 0; i < X_test->n_dims; i++) {
        printf("%d ", X_test->shape[i]);
    }
    printf("\n");

    for (int i = 0; i < 784; i++) {
        printf("%04f ", X_test->data[i]);
    }
}

