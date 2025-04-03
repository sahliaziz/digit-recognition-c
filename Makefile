# Compiler and flags
CC = gcc
CFLAGS = -Wall -g -Wextra -Wfatal-errors -O2 -I./include -lm

# Source files
SRCS = src/main.c src/tensor.c src/neural_network.c
OBJS = $(SRCS:.c=.o)

# Output executable
TARGET = run

# Default target
all: $(TARGET)

# Link the program
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(CFLAGS)

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
