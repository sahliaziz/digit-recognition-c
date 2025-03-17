#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <byteswap.h>
#include "load_mnist.h"


int main() {
    FILE *fp = fopen("train-images.idx3-ubyte", "rb");
    if (!fp) {
        perror("Error opening file");
        return 1;
    }

    struct idx *id = malloc(sizeof(struct idx));
    if (!id) {
        perror("Memory allocation failed");
        fclose(fp);
        return 1;
    }

    // Read the first 8 bytes: zero_bytes, data_type, and n_dims
    fread(&id->zero_bytes, sizeof(uint16_t), 1, fp);
    fread(&id->data_type, sizeof(uint8_t), 1, fp);
    fread(&id->n_dims, sizeof(uint8_t), 1, fp);

    // Allocate memory for dimension sizes
    id->dim_size = malloc(id->n_dims * sizeof(uint32_t));
    if (!id->dim_size) {
        perror("Memory allocation failed");
        free(id);
        fclose(fp);
        return 1;
    }

    // Read the dimensions
    uint32_t to_swap;
    for (int i = 0; i < id->n_dims; i++) {
        fread(&to_swap, sizeof(uint32_t), 1, fp);
        id->dim_size[i] = __bswap_32(to_swap);
    }

    // First dimension represents the number of images
    int n_examples = id->dim_size[0];

    // Allocate memory for image data (28x28 pixels per image)
    id->data = malloc(n_examples * 28 * 28 * sizeof(uint8_t));
    if (!id->data) {
        perror("Memory allocation failed");
        free(id->dim_size);
        free(id);
        fclose(fp);
        return 1;
    }

    // Read all image data at once
    fread(id->data, 1, n_examples * 28 * 28, fp);

    // Print some information for verification
    printf("Zero bytes: %d\n", id->zero_bytes);
    printf("Data type: %d\n", id->data_type);
    printf("Number of dimensions: %d\n", id->n_dims);

    printf("Dimensions: ");
    for (int i = 0; i < id->n_dims; i++) {
        printf("%d ", id->dim_size[i]);
    }
    printf("\n");

    // Print first image pixels for debugging (optional)
    for (int j = 0; j < 28 * 28; j++) {
        printf("%02X ", id->data[j]);
        if ((j + 1) % 28 == 0) printf("\n");  // New line every 28 pixels
    }

    // Cleanup
    free(id->data);
    free(id->dim_size);
    free(id);
    fclose(fp);

    return 0;
}
