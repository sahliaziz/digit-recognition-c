typedef struct {
    uint32_t size;
    uint8_t n_dims;
    uint32_t *shape;
    float *data;
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
Tensor *scale(Tensor *tensor, float factor);
Tensor *matrix_add_bias(Tensor *matrix, Tensor *bias);
float cross_entropy_loss(Tensor *labels, Tensor *predictions);
void evaluate_model(Tensor *X, Tensor *Y, Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2);
void show_tensor(Tensor *tensor);