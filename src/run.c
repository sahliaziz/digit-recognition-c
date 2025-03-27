#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "tensor.h"

int main() {
    srand(1337);
    Tensor *X_train = scale(idx_to_tensor("train-images.idx3-ubyte"), 1.0 / 255.0);
    Tensor *Y_train = idx_to_tensor("train-labels.idx1-ubyte");

    reshape_tensor(X_train, 60000, 784);
    reshape_tensor(Y_train, 60000, 1);

    Tensor *W1 = scale(random_tensor(784, 128), sqrt(2.0 / 784));
    Tensor *b1 = scale(random_tensor(128, 1), 0.01);
    Tensor *W2 = scale(random_tensor(128, 10), sqrt(2.0 / 128));
    Tensor *b2 = scale(random_tensor(10, 1), 0.01);

    Tensor *Y_enc = one_hot(Y_train);

    for (int i = 0; i < 1; i++) {
        // Forward pass
        Tensor *Z1 = matrix_add_bias(matmul(X_train, W1), b1);
        Tensor *A1 = ReLU(Z1);
        Tensor *Z2 = matrix_add_bias(matmul(A1, W2), b2);
        Tensor *Y_pred = softmax(Z2);

        // Backward pass
        Tensor *d_pred = add(Y_enc, scale(Y_pred, -1));
        Tensor *d_W2 = matmul(transpose(A1), d_pred);
        Tensor *d_b2 = sum(d_pred, 0);
        reshape_tensor(d_b2, 10, 1);
        Tensor *d_A1 = matmul(d_pred, transpose(W2));
        for (uint32_t i = 0; i < Z1->size; i++)
        {
            if (Z1->data[i] <= 0){
                d_A1->data[i] = 0;
            }
        }
        Tensor *d_W1 = matmul(transpose(X_train), d_A1);
        Tensor *d_b1 = sum(d_A1, 0);
        reshape_tensor(d_b1, 128, 1);

        // Update weights and biases
        W1 = add(W1, scale(d_W1, -0.005));
        b1 = add(b1, scale(d_b1, -0.005));
        W2 = add(W2, scale(d_W2, -0.005));
        b2 = add(b2, scale(d_b2, -0.005));

        // Free up some memory
        free_tensor(Z1);
        free_tensor(A1);
        free_tensor(Z2);
        free_tensor(Y_pred);
        free_tensor(d_pred);
        free_tensor(d_W2);
        free_tensor(d_b2);
        free_tensor(d_A1);
        free_tensor(d_W1);
        free_tensor(d_b1);

        //evaluate_model(X_train, Y_train, W1, b1, W2, b2);

        show_tensor(Y_pred);

        //printf("Loss: %f\n", cross_entropy_loss(Y_train, Y_pred));
    }

    evaluate_model(X_train, Y_train, W1, b1, W2, b2);

    free_tensor(X_train);
    free_tensor(Y_train);
    free_tensor(W1);
    free_tensor(b1);
    free_tensor(W2);
    free_tensor(b2);
    free_tensor(Y_enc);

    return 0;
}