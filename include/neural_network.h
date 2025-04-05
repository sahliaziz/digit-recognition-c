#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "tensor.h"

typedef struct {
    uint16_t input_size;
    uint16_t hidden_size;
    uint16_t output_size;
    Tensor *W1;
    Tensor *b1;
    Tensor *W2;
    Tensor *b2;
} NeuralNetwork;

NeuralNetwork* nn_create(uint16_t input_size, uint16_t hidden_size, uint16_t output_size);
void nn_free(NeuralNetwork *nn);
/* static Tensor* nn_forward_pass(const NeuralNetwork *nn, const Tensor *X, 
                               Tensor **Z1_out, Tensor **A1_out); */
void nn_train(NeuralNetwork *nn, const Tensor *X, const Tensor *Y, 
              float learning_rate);
void nn_predict(const NeuralNetwork *nn, const Tensor *X);
float nn_evaluate_model(NeuralNetwork *nn, Tensor *X, Tensor *Y);
void nn_save_model(const NeuralNetwork *nn, const char *filename);
NeuralNetwork* nn_load_model(const char *filename);

#endif