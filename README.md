# Digit Recognition in C

A neural network implementation for MNIST digit recognition, written in C.

## Building

1. Clone this repository:
```bash
git clone https://github.com/sahliaziz/digit-recognition-c
cd digit-recognition-c
```

2. Download and extract the dataset:
```bash
mkdir data
cd data
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

3. Compile the program:
```bash
make
```

## Running

```bash
./run
```

## Project Structure

- `include/`: Header files
- `src/`: Source files
- `data/`: MNIST dataset files
- `examples/`: Example programs
