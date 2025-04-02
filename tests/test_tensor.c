#include "tensor.h"
#include <assert.h>
#include <stdio.h>

void test_tensor_creation() {
    // Basic tensor creation test
    uint32_t shape[] = {2, 2};
    Tensor* t = tensor_create(2, shape);
    assert(t != NULL);
    tensor_free(t);
}

int main() {
    test_tensor_creation();
    printf("All tensor tests passed!\n");
    return 0;
}
