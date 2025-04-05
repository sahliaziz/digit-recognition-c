#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <byteswap.h>
#include <string.h>
#include "tensor.h"

// Helper functions
static void* tensor_malloc(size_t size, const char* error_msg) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed: %s\n", error_msg);
        return NULL;
    }
    return ptr;
}

static Tensor* tensor_create_empty(void) {
    return (Tensor*)tensor_malloc(sizeof(Tensor), "tensor structure");
}

static bool tensor_allocate_shape(Tensor* tensor, uint8_t n_dims) {
    tensor->shape = (uint32_t*)tensor_malloc(n_dims * sizeof(uint32_t), "tensor shape");
    return tensor->shape != NULL;
}

static bool tensor_allocate_data(Tensor* tensor, uint32_t size) {
    tensor->data = (float*)tensor_malloc(size * sizeof(float), "tensor data");
    return tensor->data != NULL;
}

static bool tensor_check_compatibility(const Tensor* t1, const Tensor* t2, const char* op) {
    if (!t1 || !t2) {
        fprintf(stderr, "%s: NULL tensor(s)\n", op);
        return false;
    }
    if (t1->size != t2->size) {
        fprintf(stderr, "%s: Incompatible tensor sizes\n", op);
        return false;
    }
    return true;
}

// Core tensor operations
Tensor* tensor_create(uint8_t n_dims, const uint32_t* shape) {
    if (!shape) return NULL;
    
    Tensor* tensor = tensor_create_empty();
    if (!tensor) return NULL;

    tensor->n_dims = n_dims;
    if (!tensor_allocate_shape(tensor, n_dims)) {
        tensor_free(tensor);
        return NULL;
    }

    uint32_t size = 1;
    for (uint8_t i = 0; i < n_dims; i++) {
        tensor->shape[i] = shape[i];
        size *= shape[i];
    }
    tensor->size = size;

    if (!tensor_allocate_data(tensor, size)) {
        tensor_free(tensor);
        return NULL;
    }

    return tensor;
}

Tensor* tensor_create_from_idx(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return NULL;
    }

    uint16_t zero_bytes;
    uint8_t data_type;
    if (fread(&zero_bytes, sizeof(uint16_t), 1, f) != 1 || zero_bytes != 0) {
        fprintf(stderr, "Invalid IDX file format\n");
        fclose(f);
        return NULL;
    }

    fread(&data_type, sizeof(uint8_t), 1, f);
    
    Tensor* tensor = tensor_create_empty();
    if (!tensor) {
        fclose(f);
        return NULL;
    }

    fread(&tensor->n_dims, sizeof(uint8_t), 1, f);
    if (!tensor_allocate_shape(tensor, tensor->n_dims)) {
        tensor_free(tensor);
        fclose(f);
        return NULL;
    }

    uint32_t size = 1;
    for (uint8_t i = 0; i < tensor->n_dims; i++) {
        uint32_t dim;
        fread(&dim, sizeof(uint32_t), 1, f);
        tensor->shape[i] = __bswap_32(dim);
        size *= tensor->shape[i];
    }
    tensor->size = size;

    if (!tensor_allocate_data(tensor, size)) {
        tensor_free(tensor);
        fclose(f);
        return NULL;
    }

    // Read data as uint8_t and convert to float
    uint8_t* temp_buf = (uint8_t*)tensor_malloc(size, "temporary buffer");
    if (!temp_buf) {
        tensor_free(tensor);
        fclose(f);
        return NULL;
    }

    fread(temp_buf, sizeof(uint8_t), size, f);
    for (uint32_t i = 0; i < size; i++) {
        tensor->data[i] = (float)temp_buf[i];
    }

    free(temp_buf);
    fclose(f);
    return tensor;
}

void tensor_free(Tensor* tensor) {
    if (tensor) {
        free(tensor->shape);
        free(tensor->data);
        free(tensor);
    }
}

Tensor* tensor_copy(const Tensor* tensor) {
    if (!tensor) return NULL;

    Tensor* result = tensor_create(tensor->n_dims, tensor->shape);
    if (!result) return NULL;

    for (uint32_t i = 0; i < tensor->size; i++) {
        result->data[i] = tensor->data[i];
    }

    return result;
}

bool tensor_reshape(Tensor* tensor, uint32_t n, uint32_t m) {
    if (!tensor || n * m != tensor->size) {
        fprintf(stderr, "Invalid reshape dimensions\n");
        return false;
    }

    uint32_t* new_shape_ptr = (uint32_t*)tensor_malloc(2 * sizeof(uint32_t), "reshape");
    if (!new_shape_ptr) return false;

    free(tensor->shape);
    tensor->shape = new_shape_ptr;
    tensor->n_dims = 2;
    tensor->shape[0] = n;
    tensor->shape[1] = m;

    return true;
}

Tensor* tensor_mult(const Tensor* a, const Tensor* b) {
    if (!a || !b || a->n_dims != 2 || b->n_dims != 2 || a->shape[1] != b->shape[0]) {
        fprintf(stderr, "Invalid matrices for multiplication\n");
        return NULL;
    }

    uint32_t shape[2] = {a->shape[0], b->shape[1]};
    Tensor* result = tensor_create(2, shape);
    if (!result) return NULL;

    for (uint32_t i = 0; i < a->shape[0]; i++) {
        for (uint32_t j = 0; j < b->shape[1]; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < a->shape[1]; k++) {
                sum += a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
            }
            result->data[i * b->shape[1] + j] = sum;
        }
    }

    return result;
}

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    if (!tensor_check_compatibility(a, b, "Addition")) return NULL;

    Tensor* result = tensor_create(a->n_dims, a->shape);
    if (!result) return NULL;

    for (uint32_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}

Tensor* tensor_scale(const Tensor* tensor, float factor) {
    if (!tensor) return NULL;

    Tensor* result = tensor_copy(tensor);
    if (!result) return NULL;

    for (uint32_t i = 0; i < tensor->size; i++) {
        result->data[i] *= factor;
    }

    return result;
}

Tensor* tensor_relu(const Tensor* tensor) {
    if (!tensor) return NULL;

    Tensor* result = tensor_create(tensor->n_dims, tensor->shape);
    if (!result) return NULL;

    for (uint32_t i = 0; i < tensor->size; i++) {
        result->data[i] = tensor->data[i] > 0 ? tensor->data[i] : 0;
    }

    return result;
}

Tensor* tensor_softmax(const Tensor* tensor) {
    if (!tensor || tensor->n_dims != 2) {
        fprintf(stderr, "Invalid tensor for softmax\n");
        return NULL;
    }

    Tensor* result = tensor_copy(tensor);
    if (!result) return NULL;

    for (uint32_t i = 0; i < tensor->shape[0]; i++) {
        // Find max for numerical stability
        float max_val = result->data[i * tensor->shape[1]];
        for (uint32_t j = 1; j < tensor->shape[1]; j++) {
            float val = result->data[i * tensor->shape[1] + j];
            if (val > max_val) max_val = val;
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (uint32_t j = 0; j < tensor->shape[1]; j++) {
            float val = exp(result->data[i * tensor->shape[1] + j] - max_val);
            result->data[i * tensor->shape[1] + j] = val;
            sum += val;
        }

        // Normalize
        for (uint32_t j = 0; j < tensor->shape[1]; j++) {
            result->data[i * tensor->shape[1] + j] /= sum;
        }
    }

    return result;
}

Tensor* tensor_one_hot(const Tensor* labels) {
    if (!labels) return NULL;

    uint32_t shape[2] = {labels->size, 10};
    Tensor* result = tensor_create(2, shape);
    if (!result) return NULL;

    for (uint32_t i = 0; i < labels->size; i++) {
        for (uint32_t j = 0; j < 10; j++) {
            result->data[i * 10 + j] = (j == (uint32_t)labels->data[i]) ? 1.0f : 0.0f;
        }
    }

    return result;
}

Tensor* tensor_argmax(const Tensor* tensor) {
    if (!tensor || tensor->n_dims != 2) {
        fprintf(stderr, "Invalid tensor for argmax\n");
        return NULL;
    }

    uint32_t shape[2] = {tensor->shape[0], 1};
    Tensor* result = tensor_create(2, shape);
    if (!result) return NULL;

    for (uint32_t i = 0; i < tensor->shape[0]; i++) {
        float max_val = tensor->data[i * tensor->shape[1]];
        uint32_t max_idx = 0;

        for (uint32_t j = 1; j < tensor->shape[1]; j++) {
            float val = tensor->data[i * tensor->shape[1] + j];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        result->data[i] = (float)max_idx;
    }

    return result;
}

float tensor_cross_entropy_loss(const Tensor* labels, const Tensor* predictions) {
    if (!labels || !predictions || labels->shape[0] != predictions->shape[0]) {
        fprintf(stderr, "Invalid tensors for cross entropy\n");
        return INFINITY;
    }

    float loss = 0.0f;
    const float epsilon = 1e-10f;

    for (uint32_t i = 0; i < labels->size; i++) {
        uint32_t true_label = (uint32_t)labels->data[i];
        float pred = predictions->data[i * predictions->shape[1] + true_label];
        loss -= log(pred + epsilon);
    }

    return loss / labels->shape[0];
}

void tensor_print(const Tensor* tensor) {
    if (!tensor) {
        fprintf(stderr, "NULL tensor\n");
        return;
    }

    printf("Tensor shape: ");
    for (uint8_t i = 0; i < tensor->n_dims; i++) {
        printf("%u ", tensor->shape[i]);
    }
    printf("\n");

    if (tensor->n_dims == 2) {
        for (uint32_t i = 0; i < 5; i++) { // tensor->shape[0]
            for (uint32_t j = 0; j < fmin(tensor->shape[1], 10); j++) {
                printf("%8.4f ", tensor->data[i * tensor->shape[1] + j]);
            }
            printf("\n");
        }
    } else {
        printf("Data: ");
        for (uint32_t i = 0; i < tensor->size && i < 10; i++) {
            printf("%8.4f ", tensor->data[i]);
        }
        if (tensor->size > 10) printf("...");
        printf("\n");
    }
}

Tensor *tensor_sum(const Tensor *tensor, int8_t axis) {
    if (axis < 0 || axis >= tensor->n_dims) {
        perror("Invalid axis");
        return NULL;
    }

    Tensor *result = malloc(sizeof(Tensor));
    if (!result)
    {
        perror("Memory allocation failed");
        return NULL;
    }

    result->n_dims = tensor->n_dims;
    result->shape = malloc(result->n_dims * sizeof(uint32_t));
    if (!result->shape)
    {
        perror("Memory allocation failed");
        free(result);
        return NULL;
    }

    for (uint32_t i = 0; i < tensor->n_dims; i++)
    {
        result->shape[i] = tensor->shape[i];
    }
    result->shape[axis] = 1;

    uint32_t new_size = 1;
    for (uint32_t i = 0; i < result->n_dims; i++)
    {
        new_size *= result->shape[i];
    }
    result->size = new_size;

    result->data = malloc(result->size * sizeof(float));
    if (!result->data)
    {
        perror("Memory allocation failed");
        free(result->shape);
        free(result);
        return NULL;
    }

    uint32_t stride = 1;
    for (int8_t i = tensor->n_dims - 1; i > axis; i--)
    {
        stride *= tensor->shape[i];
    }

    for (uint32_t i = 0; i < result->size; i++)
    {
        result->data[i] = 0;
        for (uint32_t j = 0; j < tensor->shape[axis]; j++)
        {
            result->data[i] += tensor->data[(i / stride * tensor->shape[axis] + j) * stride + i % stride];
        }
    }

    return result;
}

Tensor *tensor_create_random(uint32_t n, uint32_t m) {
    Tensor *tensor = tensor_create(2, (uint32_t[]){n, m});
    if (!tensor) return NULL;

    for (uint32_t i = 0; i < tensor->size; i++) {
        tensor->data[i] = (float)(rand() % 100) / 100.0f;
    }

    return tensor;
}

Tensor *tensor_transpose(const Tensor *tensor) {
    if (!tensor) return NULL;

    Tensor *result = tensor_create(tensor->n_dims, tensor->shape);
    if (!result) return NULL;

    uint32_t temp = result->shape[0];
    result->shape[0] = result->shape[1];
    result->shape[1] = temp;

    for (uint32_t i = 0; i < tensor->shape[0]; i++) {
        for (uint32_t j = 0; j < tensor->shape[1]; j++) {
            result->data[j * tensor->shape[0] + i] = tensor->data[i * tensor->shape[1] + j];
        }
    }

    return result;
}

Tensor *tensor_add_bias(const Tensor *matrix, const Tensor *bias) {
    if (!matrix || !bias || matrix->n_dims != 2 || bias->n_dims != 2 || matrix->shape[1] != bias->shape[0]) {
        fprintf(stderr, "Invalid matrix or bias for addition\n");
        return NULL;
    }

    Tensor *result = tensor_copy(matrix);
    if (!result) return NULL;
    
    for (uint32_t i = 0; i < matrix->shape[0]; i++) {
        for (uint32_t j = 0; j < matrix->shape[1]; j++) {
            result->data[i * matrix->shape[1] + j] = matrix->data[i * matrix->shape[1] + j] + bias->data[j];
        }
    }

    return result;
}

Tensor **tensor_batch(Tensor *tensor, uint32_t batch_size, uint32_t *n_batches) {
    if (batch_size == 0 || tensor->shape[0] % batch_size != 0) {
        perror("Invalid batch size");
        return NULL;
    }

    *n_batches = tensor->shape[0] / batch_size;
    Tensor **batches = malloc(*n_batches * sizeof(Tensor *));
    if (!batches) {
        perror("Memory allocation failed");
        return NULL;
    }

    for (uint32_t i = 0; i < *n_batches; i++) {
        batches[i] = malloc(sizeof(Tensor));
        if (!batches[i]) {
            perror("Memory allocation failed");
            for (uint32_t j = 0; j < i; j++)
                tensor_free(batches[j]);
            free(batches);
            return NULL;
        }

        batches[i]->n_dims = tensor->n_dims;
        batches[i]->shape = malloc(tensor->n_dims * sizeof(uint32_t));
        if (!batches[i]->shape){
            perror("Memory allocation failed");
            for (uint32_t j = 0; j <= i; j++)
                tensor_free(batches[j]);
            free(batches);
            return NULL;
        }

        memcpy(batches[i]->shape, tensor->shape, tensor->n_dims * sizeof(uint32_t));
        batches[i]->shape[0] = batch_size;
        batches[i]->size = batch_size * tensor->shape[1];
        batches[i]->data = malloc(batches[i]->size * sizeof(float));
        if (!batches[i]->data) {
            perror("Memory allocation failed");
            for (uint32_t j = 0; j <= i; j++)
                tensor_free(batches[j]);
            free(batches);
            return NULL;
        }

        memcpy(batches[i]->data, tensor->data + i * batches[i]->size, batches[i]->size * sizeof(float));
    }

    return batches;
}