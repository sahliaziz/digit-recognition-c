#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stdbool.h>

// Type definitions
typedef struct {
    uint32_t size;
    uint8_t n_dims;
    uint32_t *shape;
    float *data;
} Tensor;

// Tensor creation and destruction
Tensor *tensor_create_from_idx(const char *filename);
Tensor *tensor_create_random(uint32_t n, uint32_t m);
Tensor *tensor_copy(const Tensor *tensor);
void tensor_free(Tensor *tensor);

// Basic tensor operations
bool tensor_reshape(Tensor *tensor, uint32_t n, uint32_t m);
Tensor *tensor_transpose(const Tensor *tensor);
Tensor *tensor_add(const Tensor *tensor1, const Tensor *tensor2);
Tensor *tensor_scale(const Tensor *tensor, float factor);
Tensor *tensor_sum(const Tensor *tensor, int8_t axis);

// Neural network specific operations
Tensor *tensor_mult(const Tensor *matrix1, const Tensor *matrix2);
Tensor *tensor_add_bias(const Tensor *matrix, const Tensor *bias);
Tensor *tensor_relu(const Tensor *tensor);
Tensor *tensor_softmax(const Tensor *tensor);
Tensor *tensor_one_hot(const Tensor *labels);
Tensor *tensor_argmax(const Tensor *tensor);

// Model evaluation
float tensor_cross_entropy_loss(const Tensor *labels, const Tensor *predictions);

// Debug utilities
void tensor_print(const Tensor *tensor);

#endif // TENSOR_H