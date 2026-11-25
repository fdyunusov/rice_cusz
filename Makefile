NVCC = nvcc
ARCH = -arch=sm_75
CFLAGS = -O3 -std=c++17 $(ARCH)

all: rice_cusz

rice_cusz: main.cu dual_quant.cuh rice.cuh common.cuh
	$(NVCC) $(CFLAGS) main.cu -o rice_cusz

clean:
	rm -f rice_cusz
