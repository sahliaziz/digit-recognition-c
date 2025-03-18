typedef struct {
    uint8_t data_type;
    uint32_t size;
    uint8_t n_dims;
    uint32_t *dim_size;
    uint8_t *data;
} IDX;

IDX *load_from_idx(char*);