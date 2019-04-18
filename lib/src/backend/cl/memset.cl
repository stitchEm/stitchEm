// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

kernel void memsetToZero(__global unsigned char *buffer, unsigned settingSize) {
  const size_t index = get_global_id(0);
  if (index < settingSize) {
    buffer[index] = (unsigned char)0;
  }
}
