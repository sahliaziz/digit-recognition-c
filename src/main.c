#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"
#include "neural_network.h"

#define LEARNING_RATE 0.005
#define HIDDEN_SIZE 128
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define NUM_EPOCHS 3

int main() {
    // Set random seed for reproducibility
    srand(1337);

    // Load and preprocess data
    Tensor *X_train = tensor_create_from_idx("data/train-images.idx3-ubyte");
    Tensor *Y_train = tensor_create_from_idx("data/train-labels.idx1-ubyte");
    
    if (!X_train || !Y_train) {
        fprintf(stderr, "Failed to load training data\n");
        return 1;
    }

    // Preprocess data
    tensor_scale(X_train, 1.0 / 255.0);
    tensor_reshape(X_train, 60000, INPUT_SIZE);
    tensor_reshape(Y_train, 60000, 1);

    // Create and initialize neural network
    NeuralNetwork *nn = nn_create(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    if (!nn) {
        fprintf(stderr, "Failed to create neural network\n");
        tensor_free(X_train);
        tensor_free(Y_train);
        return 1;
    }

    // Train the network
    nn_train(nn, X_train, Y_train, LEARNING_RATE, NUM_EPOCHS);

    // Evaluate the model
    nn_predict(nn, X_train);

    // Cleanup
    nn_free(nn);
    tensor_free(X_train);
    tensor_free(Y_train);

    return 0;
}