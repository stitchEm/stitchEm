// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

/**
 * This kernel computes the OR of all pixels in each row, and pouts the result in
 * colHasImage
 * FIXME do it with parallel reduction
 */
void kernel vertOrKernel(const global unsigned int* contrib, global unsigned int* colHasImage, unsigned panoWidth,
                         unsigned panoHeight) {
  unsigned col = (unsigned)get_global_id(0);

  if (col < panoWidth) {
    unsigned int accum = 0;
    for (unsigned row = 0; row < panoHeight; ++row) {
      accum |= contrib[panoWidth * row + col];
    }
    colHasImage[col] = accum;
  }
}

void kernel horizOrKernel(const global unsigned int* contrib, global unsigned int* rowHasImage, unsigned panoWidth,
                          unsigned panoHeight) {
  unsigned row = (unsigned)get_global_id(0);

  const global unsigned int* rowp = contrib + panoWidth * row;

  if (row < panoHeight) {
    unsigned int accum = 0;
    for (unsigned col = 0; col < panoWidth; ++col) {
      accum |= rowp[col];
    }
    rowHasImage[row] = accum;
  }
}
