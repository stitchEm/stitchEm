// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

__kernel void vecAddDummy(__global float* output, __global const float* input, unsigned nbElem, float mult) {
  for (size_t i = get_global_id(0); i < (size_t)nbElem; i += get_global_size(0)) {
    output[i] = mult * input[i];
  }
}
