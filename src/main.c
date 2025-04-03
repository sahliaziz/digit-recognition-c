#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"
#include "neural_network.h"

#define LEARNING_RATE 0.001
#define HIDDEN_SIZE 128
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define NUM_EPOCHS 2

int main() {
    // Set random seed for reproducibility
    srand(1337);

    // Load and preprocess data
    Tensor *X_train = tensor_create_from_idx("data/train-images.idx3-ubyte");
    Tensor *Y_train = tensor_create_from_idx("data/train-labels.idx1-ubyte");
    Tensor *X_test = tensor_create_from_idx("data/t10k-images.idx3-ubyte");
    Tensor *Y_test = tensor_create_from_idx("data/t10k-labels.idx1-ubyte");
    
    if (!X_train || !Y_train) {
        fprintf(stderr, "Failed to load training data\n");
        return 1;
    }

    // Preprocess data
    X_train = tensor_scale(X_train, 1.0 / 255.0);
    X_test = tensor_scale(X_test, 1.0 / 255.0);
    tensor_reshape(X_train, X_train->shape[0], INPUT_SIZE);
    tensor_reshape(Y_train, Y_train->shape[0], 1);
    tensor_reshape(X_test, X_test->shape[0], INPUT_SIZE);
    tensor_reshape(Y_test, Y_test->shape[0], 1);

    uint32_t n_batches = 60000 / 100;

    Tensor **X_batch = tensor_batch(X_train, 100, &n_batches);
    Tensor **Y_batch = tensor_batch(Y_train, 100, &n_batches);

    printf("Training data loaded and preprocessed.\n");


    // Create and initialize neural network
    NeuralNetwork *nn = nn_create(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    if (!nn) {
        fprintf(stderr, "Failed to create neural network\n");
        tensor_free(X_train);
        tensor_free(Y_train);
        return 1;
    }

    printf("Neural network created with input size %d, hidden size %d, output size %d.\n",
           INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);


    // Train the network
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        printf("Epoch %d/%d\n", epoch + 1, NUM_EPOCHS);
        for (uint32_t i = 0; i < n_batches; i++) {
            nn_train(nn, X_batch[i], Y_batch[i], LEARNING_RATE);
        }
    }

    printf("Training completed.\n");


    // Evaluate the model
    float accuracy_train = nn_evaluate_model(nn, X_train, Y_train);
    float accuracy_test = nn_evaluate_model(nn, X_test, Y_test);

    printf("Model accuracy (tarining set): %.2f%%\n", accuracy_train * 100);
    printf("Model accuracy (test set): %.2f%%\n", accuracy_test * 100);


    // Cleanup
    nn_free(nn);
    tensor_free(X_train);
    tensor_free(Y_train);

    return 0;
}