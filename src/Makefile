CC = gcc
CFLAGS = -Wall -g
TARGET = run
OBJS = run.o tensor.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -lm
