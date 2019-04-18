// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

// Combining several related parameters into a struct to be passed
// to the GPU backend

typedef struct {
  int width;
  int height;
  float scaleX;
  float scaleY;
} PanoDimensions;

typedef struct {
  // global dimensions
  PanoDimensions panoDim;

  // view/region offset
  int viewTop;
  int viewLeft;

  // view size
  int viewWidth;
  int viewHeight;
} PanoRegion;
