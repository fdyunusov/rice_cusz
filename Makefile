NVCC ?= nvcc
TARGET := sz_rice

SRC := src/main.cu src/lorenzo.cu src/rice.cu
INCLUDES := -Iinclude

# Adjust -arch to your GPU if needed (e.g. sm_75, sm_86, sm_89)
NVCCFLAGS := -O3 -std=c++17 $(INCLUDES) -arch=native

.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(SRC) include/*.cuh
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) --n 1048576 --eb 1e-3 --rice-k 2 --self-test

test: run

clean:
	rm -f $(TARGET) $(TARGET).exe
