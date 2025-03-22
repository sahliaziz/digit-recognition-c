typedef struct {
    uint8_t data_type;
    uint32_t size;
    uint8_t n_dims;
    uint32_t *shape;
    double *data;
} Tensor;


Tensor *idx_to_tensor(char*);
Tensor *matmul(Tensor *matrix1, Tensor *matrix2);
Tensor *transpose(Tensor *tensor);
void free_tensor(Tensor *tensor);
double sum(Tensor *tensor);
void reshape_tensor(Tensor *tensor, uint32_t n, uint32_t m);
