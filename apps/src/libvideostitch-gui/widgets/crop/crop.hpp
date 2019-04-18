// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

/**
 * @brief A struct that represents the four values of a crop
 */
struct VS_GUI_EXPORT Crop {
  /**
   * @brief Default constructor.
   */
  Crop() : crop_left(0), crop_right(0), crop_top(0), crop_bottom(0) {}

  /**
   * @brief Contructor
   * @param crop_left The left crop value.
   * @param crop_right The right crop value.
   * @param crop_top The top crop value.
   * @param crop_bottom The bottom crop value.
   */
  Crop(int crop_left, int crop_right, int crop_top, int crop_bottom)
      : crop_left(crop_left), crop_right(crop_right), crop_top(crop_top), crop_bottom(crop_bottom) {}

  /**
   * @brief Operator equal between two crops.
   * @param other A crop to compare.
   * @return True if they are equal.
   */
  bool operator==(const Crop& other) {
    return (this->crop_left == other.crop_left && this->crop_right == other.crop_right &&
            this->crop_top == other.crop_top && this->crop_bottom == other.crop_bottom);
  }

  int crop_left;
  int crop_right;
  int crop_top;
  int crop_bottom;
};
