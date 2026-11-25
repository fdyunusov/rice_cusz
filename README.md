# SZ-Rice: GPU error-bounded compressor with Golomb–Rice backend

Class final project (COP6527): an **SZ-style** lossy compressor on CUDA whose lossless stage is **Golomb–Rice**, instead of Huffman (as in cuSZ).

All non-trivial code in this folder is original. The [cuSZ](https://arxiv.org/abs/2007.09625) paper is used only for the dual-quant / Lorenzo *algorithm idea*.

## Pipeline

1. **Prequant** \(d' = \mathrm{round}(x / (2\cdot eb))\)
2. **1D Lorenzo** prediction on \(d'\); store deltas \(\delta_i = d'_i - d'_{i-1}\)
3. **ZigZag** + **Rice(\(k\))** encode deltas on the GPU
4. Decompress: Rice decode → inclusive scan of deltas → \( \hat{x} = d' \cdot (2\cdot eb) \)

Guarantees \(\lvert x - \hat{x}\rvert \le eb\) (absolute) for finite inputs, provided `eb` is not smaller than a few float32 ULPs of the data values (e.g. around magnitude 1000, ULP ≈ 1e-4).

## Build

**Windows**

```bat
build.bat
```

**Linux / WSL**

```bash
make
./sz_rice --self-test --n 1048576 --eb 1e-3 --rice-k 2
```

If `-arch=native` fails, set e.g. `-arch=sm_86` for an RTX 30-series GPU.

## Usage

```
sz_rice --n 1048576 --eb 1e-3 --rice-k 2 --pattern smooth
sz_rice --file data.bin --eb 1e-4 --rice-k 3
sz_rice --self-test
```

| Flag | Meaning |
|------|---------|
| `--n` | Synthetic length |
| `--eb` | Absolute error bound |
| `--rice-k` | Rice parameter \(k\) (try 1–4) |
| `--pattern` | `smooth` / `noisy` / `ramp` |
| `--file` | Raw `float32` binary |
| `--self-test` | Host correctness checks first |

## Layout

```
sz_rice/
  include/common.cuh     CUDA helpers + stats
  include/lorenzo.cuh    Dual-quant + Lorenzo API
  include/rice.cuh       Rice / ZigZag API
  src/lorenzo.cu         Prediction / reconstruction kernels
  src/rice.cu            GPU Rice encode/decode
  src/main.cu            CLI, timing, verification
  STEP_BY_STEP.md        How to finish report/demo
  Makefile / build.bat
```

## Team split (suggested)

- Person A: Lorenzo / error-bound experiments / datasets  
- Person B: Rice bit-packing / timing tables  
- Both: report + demo  

## Citation note for the report

Tian et al., *cuSZ: An Efficient GPU-Based Error-Bounded Lossy Compression Framework for Scientific Data*, PACT 2020.  
We replace the Huffman lossless backend with a GPU Golomb–Rice codec.
