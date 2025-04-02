#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "tensor.h"

typedef struct {
    Tensor *W1;
    Tensor *b1;
    Tensor *W2;
    Tensor *b2;
} NeuralNetwork;

NeuralNetwork* nn_create(uint32_t input_size, uint32_t hidden_size, uint32_t output_size);
void nn_free(NeuralNetwork *nn);
void nn_train(NeuralNetwork *nn, const Tensor *X, const Tensor *Y, 
              float learning_rate, int epochs);
void nn_predict(const NeuralNetwork *nn, const Tensor *X);
float nn_evaluate_model(NeuralNetwork *nn, Tensor *X, Tensor *Y);

#endif