#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <byteswap.h>
#include "load_mnist.h"


IDX * load_from_idx(char* filename) {
    uint16_t zero_bytes;
    uint32_t to_swap;
    uint32_t size = 1;

    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Error opening file");
        fclose(f);
        return NULL;
    }

    IDX *idx = malloc(sizeof(IDX));

    if (!idx) {
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
        fclose(f);
        return NULL;
    }

    fread(&idx->data_type, sizeof(uint8_t), 1, f);
    fread(&idx->n_dims, sizeof(uint8_t), 1, f);

    // Allocate memory for dimension sizes
    idx->dim_size = malloc(idx->n_dims * sizeof(uint32_t));
    if (!idx->dim_size) {
        perror("Memory allocation failed");
        free(idx);
        fclose(f);
        return NULL;
    }

    // Reading dimensions
    for (int i = 0; i < idx->n_dims; i++) {
        fread(&to_swap, sizeof(uint32_t), 1, f);
        idx->dim_size[i] = __bswap_32(to_swap);
        size *= idx->dim_size[i];
    }

    // First dimension represents the number of images
    uint32_t n_examples = idx->dim_size[0];
    idx->size = size;


    // Allocating memory for image data (28x28 pixels per image)
    idx->data = malloc(size * sizeof(uint8_t));
    if (!idx->data) {
        perror("Memory allocation failed");
        free(idx->dim_size);
        free(idx);
        fclose(f);
        return NULL;
    }

    // Read all image data at once
    fread(idx->data, 1, size, f);
    fclose(f);

    return idx;
}


int main() {
    IDX * X_test = load_from_idx("train-images.idx3-ubyte");

    // Printing some information for debugging
    printf("Data type: %d\n", X_test->data_type);
    printf("Number of dimensions: %d\n", X_test->n_dims);
    printf("Number of pixels: %d\n", X_test->size);

    printf("Dimensions: ");
    for (int i = 0; i < X_test->n_dims; i++) {
        printf("%d ", X_test->dim_size[i]);
    }
    printf("\n");
}

