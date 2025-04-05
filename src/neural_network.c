#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "neural_network.h"

NeuralNetwork *nn_create(uint16_t input_size, uint16_t hidden_size, uint16_t output_size) {
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    if (!nn) {
        fprintf(stderr, "Failed to allocate neural network\n");
        return NULL;
    }

    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;

    printf("%d\n", nn->input_size);
    printf("%d\n", nn->hidden_size); 
    printf("%d\n", nn->output_size);

    // Initialize weights with He initialization
    nn->W1 = tensor_scale(tensor_create_random(input_size, hidden_size), 
                         sqrt(2.0 / input_size));
    nn->b1 = tensor_scale(tensor_create_random(hidden_size, 1), 0.01);
    nn->W2 = tensor_scale(tensor_create_random(hidden_size, output_size), 
                         sqrt(2.0 / hidden_size));
    nn->b2 = tensor_scale(tensor_create_random(output_size, 1), 0.01);

    if (!nn->W1 || !nn->b1 || !nn->W2 || !nn->b2) {
        fprintf(stderr, "Failed to initialize neural network weights\n");
        nn_free(nn);
        return NULL;
    }

    return nn;
}

void nn_free(NeuralNetwork *nn) {
    if (nn) {
        tensor_free(nn->W1);
        tensor_free(nn->b1);
        tensor_free(nn->W2);
        tensor_free(nn->b2);
        free(nn);
    }
}

static Tensor *nn_forward_pass(const NeuralNetwork *nn, const Tensor *X, 
                              Tensor **Z1_out, Tensor **A1_out) {
    *Z1_out = tensor_add_bias(tensor_mult(X, nn->W1), nn->b1);
    *A1_out = tensor_relu(*Z1_out);
    Tensor *Z2 = tensor_add_bias(tensor_mult(*A1_out, nn->W2), nn->b2);
    return tensor_softmax(Z2);
}

void nn_train(NeuralNetwork *nn, const Tensor *X, const Tensor *Y, 
              float learning_rate) {
    Tensor *Y_enc = tensor_one_hot(Y);
    if (!Y_enc) return;

    // Forward pass
    Tensor *Z1, *A1;
    Tensor *Y_pred = nn_forward_pass(nn, X, &Z1, &A1);
    if (!Y_pred) {
        tensor_free(Y_enc);
        return;
    }

    // Backward pass
    Tensor *d_pred = tensor_add(Y_pred, tensor_scale(Y_enc, -1));
    Tensor *d_W2 = tensor_mult(tensor_transpose(A1), d_pred);
    Tensor *d_b2 = tensor_sum(d_pred, 0);
    tensor_reshape(d_b2, 10, 1);

    Tensor *d_A1 = tensor_mult(d_pred, tensor_transpose(nn->W2));
    // Apply ReLU derivative
    for (uint32_t i = 0; i < Z1->size; i++) {
        if (Z1->data[i] <= 0) {
            d_A1->data[i] = 0;
        }
    }

    Tensor *d_W1 = tensor_mult(tensor_transpose(X), d_A1);
    Tensor *d_b1 = tensor_sum(d_A1, 0);
    tensor_reshape(d_b1, 128, 1);

    // Update weights and biases
    nn->W1 = tensor_add(nn->W1, tensor_scale(d_W1, -learning_rate));
    nn->b1 = tensor_add(nn->b1, tensor_scale(d_b1, -learning_rate));
    nn->W2 = tensor_add(nn->W2, tensor_scale(d_W2, -learning_rate));
    nn->b2 = tensor_add(nn->b2, tensor_scale(d_b2, -learning_rate));

    // Calculate and print loss
    float loss = tensor_cross_entropy_loss(Y, Y_pred);
    printf("Loss: %f\n", loss);

    // Free temporary tensors
    tensor_free(Z1);
    tensor_free(A1);
    tensor_free(Y_pred);
    tensor_free(d_pred);
    tensor_free(d_W2);
    tensor_free(d_b2);
    tensor_free(d_A1);
    tensor_free(d_W1);
    tensor_free(d_b1);
    tensor_free(Y_enc);
}

void nn_predict(const NeuralNetwork *nn, const Tensor *X) {
    Tensor *Z1, *A1;
    Tensor *predictions = nn_forward_pass(nn, X, &Z1, &A1);
    if (!predictions) return;

    // Convert to class predictions
    Tensor *classes = tensor_argmax(predictions);
    
    // Print first few predictions
    printf("First 10 predictions:\n");
    for (uint32_t i = 0; i < 10 && i < classes->size; i++) {
        printf("%d ", (int)classes->data[i]);
    }
    printf("\n");

    // Free temporary tensors
    tensor_free(Z1);
    tensor_free(A1);
    tensor_free(predictions);
    tensor_free(classes);
}

float nn_evaluate_model(NeuralNetwork *nn, Tensor *X, Tensor *Y) {
    // Forward pass
    Tensor *Z1 = tensor_mult(X, nn->W1);
    Tensor *A1 = tensor_relu(Z1);
    Tensor *Z2 = tensor_mult(A1, nn->W2);
    Tensor *Y_pred = tensor_softmax(Z2);

    // Calculate accuracy
    Tensor *pred_labels = tensor_argmax(Y_pred);

    uint32_t correct = 0;
    for (uint32_t i = 0; i < Y->shape[0]; i++) {
        if (pred_labels->data[i] == Y->data[i]){
            correct++;
        }
    }

    // Free allocated memory
    tensor_free(Z1);
    tensor_free(A1);
    tensor_free(Z2);
    tensor_free(Y_pred);
    tensor_free(pred_labels);

    float accuracy = (float)correct / Y->shape[0];
    
    return accuracy;
}

void nn_save_model(const NeuralNetwork *nn, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for saving model\n");
        return;
    }

    printf("Saving model: input_size=%u, hidden_size=%u, output_size=%u\n", nn->input_size, nn->hidden_size, nn->output_size);

    // Save layer sizes
    fwrite(&nn->input_size, sizeof(uint16_t), 1, file);
    fwrite(&nn->hidden_size, sizeof(uint16_t), 1, file);
    fwrite(&nn->output_size, sizeof(uint16_t), 1, file);

    // Save weights and biases
    fwrite(nn->W1->data, sizeof(float), nn->W1->size, file);
    fwrite(nn->b1->data, sizeof(float), nn->b1->size, file);
    fwrite(nn->W2->data, sizeof(float), nn->W2->size, file);
    fwrite(nn->b2->data, sizeof(float), nn->b2->size, file);

    fclose(file);
}

NeuralNetwork *nn_load_model(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file for loading model\n");
        return NULL;
    }

    uint16_t input_size, hidden_size, output_size;
    // Read layer sizes

    fread(&input_size, sizeof(uint16_t), 1, file);
    fread(&hidden_size, sizeof(uint16_t), 1, file);
    fread(&output_size, sizeof(uint16_t), 1, file);

    NeuralNetwork *nn = nn_create(input_size, hidden_size, output_size);

    if (!nn) {
        fprintf(stderr, "Failed to create neural network for loading model\n");
        fclose(file);
        return NULL;
    }

    // Load weights and biases
    fread(nn->W1->data, sizeof(float), nn->W1->size, file);
    fread(nn->b1->data, sizeof(float), nn->b1->size, file);
    fread(nn->W2->data, sizeof(float), nn->W2->size, file);
    fread(nn->b2->data, sizeof(float), nn->b2->size, file);

    fclose(file);
    return nn;
}