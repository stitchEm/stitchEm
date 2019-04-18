"""Generate gaussian kernels

Generates c++ code for unrolled small-support gaussian kernels.
"""
import sys

def main():
  print("#ifndef UNROLLED_GAUSSIAN_KERNEL_HPP_")
  print("#define UNROLLED_GAUSSIAN_KERNEL_HPP_\n")
  print("namespace VideoStitch {\n")
  print("namespace Core {\n")

  print("inline __device__ void addToAccumulatorsWeightedRGBA(const int32_t* &argb, int32_t &tr, int32_t &tg, int32_t &tb, int32_t &acc, int32_t weight) {")
  print("  int32_t isSolid = (*argb++) * weight;");
  print("  tr += isSolid * (*argb++);")
  print("  tg += isSolid * (*argb++);")
  print("  tb += isSolid * (*argb++);")
  print("  acc += isSolid;")
  print("}\n")
  
  for n in range(2, 14, 2):
    v = 1
    print("__device__ inline uint32_t unrolledGaussianKernel%i(const int32_t *col) {" % (n / 2))
    print("  int32_t tr = 0;")
    print("  int32_t tg = 0;")
    print("  int32_t tb = 0;")
    print("  int32_t acc = 0;")
    for i in range(n + 1):
      if i == n / 2:
        print("  int32_t isSolid = *col;")
        print("  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, %i);" % (v))
      else:
        print("  addToAccumulatorsWeightedRGBA(col, tr, tg, tb, acc, %i);" % (v))
      v = v * (n - i) / ( i + 1 )
    print("  return RGB210::pack(tr / acc, tr / acc, tr / acc, isSolid);")
    print("}\n")
  print("}\n")
  print("}\n")
  print("#endif")

if __name__ == "__main__":
    main()