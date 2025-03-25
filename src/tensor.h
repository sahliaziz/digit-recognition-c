typedef struct {
    uint32_t size;
    uint8_t n_dims;
    uint32_t *shape;
    double *data;
} Tensor;


Tensor *idx_to_tensor(char*);
Tensor *matmul(Tensor *matrix1, Tensor *matrix2);
Tensor *transpose(Tensor *tensor);
void free_tensor(Tensor *tensor);
Tensor *sum(Tensor *tensor, int8_t);
void reshape_tensor(Tensor *tensor, uint32_t n, uint32_t m);
Tensor *one_hot(Tensor *labels);
Tensor *random_tensor(uint32_t n, uint32_t m);
Tensor *argmax(Tensor *tensor);
Tensor *ReLU(Tensor *tensor);
Tensor *softmax(Tensor *tensor);
Tensor *add(Tensor *tensor1, Tensor *tensor2);
Tensor *copy_tensor(Tensor *tensor);