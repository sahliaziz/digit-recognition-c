struct idx {
    uint16_t zero_bytes;
    uint8_t data_type;
    uint8_t n_dims;
    uint32_t *dim_size;
    uint8_t *data;
};