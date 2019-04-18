// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <cstdio>

int main() {
  int devices;
  cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
    return 1;
  }

  if (devices == 0) {
    fprintf(stderr, "No CUDA device found!\n");
    return 1;
  }

  cudaDeviceProp props_1;
  cudaDeviceProp props_2;
  for (int i = 0; i < devices; ++i) {
    props_2 = props_1;
    err = cudaGetDeviceProperties(&props_1, i);
    if (err != cudaSuccess) {
      fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
      return 1;
    }
    if (i > 0 && (props_1.major != props_2.major || props_1.minor != props_2.minor)) {
      fprintf(stderr, "Multiple CUDA arch not supported at the moment\n");
      return 1;
    }
  }

  printf("CUDA compute capability: %d%d", props_1.major, props_1.minor);

  // Easiest cross-platform way to pass the number seems to be the exit code
  // nvcc --run on windows seems to print the filename through cl.exe (that can't be silenced)
  return props_1.major * 10 + props_1.minor;
}
