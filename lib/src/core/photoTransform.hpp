// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PHOTO_TRANSFORM_HPP_
#define PHOTO_TRANSFORM_HPP_

#include "backend/common/core/transformPhotoParam.hpp"

#include "coordinates.hpp"

#include <stdint.h>

namespace VideoStitch {
namespace Core {

class InputDefinition;
class PanoDefinition;

/**
 * A class that represents the photometric transformations applied between an input and the pano output.
 */
class PhotoTransform {
 public:
  /**
   * A class that holds an input's color correction parameters.
   */
  class ColorCorrectionParams {
   public:
    /**
     * Create from explicit values.
     */
    ColorCorrectionParams(double ev, double redCB, double greenCB, double blueCB)
        : ev(ev), redCB(redCB), greenCB(greenCB), blueCB(blueCB) {}

    /**
     * Create from a color multiplier so that greenCB == 1.0 if possible.
     * Note that it's not always possible (e.g. if the green component if colorMult is 0.0 while the red or blue is >
     * 0.0).
     * @param colorMult multiplier
     * @param panoEv Exposure value of the panorama
     * @param panoRedCB red compensation value of the panorama
     * @param panoGreenCB red compensation value of the panorama
     * @param panoBlueCB red compensation value of the panorama
     */
    static ColorCorrectionParams canonicalFromMultiplier(float3 colorMult, double panoEv, double panoRedCB,
                                                         double panoGreenCB, double panoBlueCB);

    /**
     * Computes the actual color multiplier (transfer function in linear space) for this correction.
     * @param panoEv Exposure value of the panorama
     */
    float3 computeColorMultiplier(double panoEv, double panoRedCB, double panoGreenCB, double panoBlueCB) const;

    /**
     * Exposure value.
     */
    const double ev;
    /**
     * Red correction.
     */
    const double redCB;
    /**
     * Green correction.
     */
    const double greenCB;
    /**
     * Blue correction.
     */
    const double blueCB;
  };

  virtual ~PhotoTransform() {}

 protected:
  explicit PhotoTransform(double idds);
  PhotoTransform() {}

  /**
   * Compute vignetting multiplier.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param uv coordinates in input space, relative to the center of the input.
   */
  double computeVignettingMult(const InputDefinition& im, const CenterCoords2& uv) const;
  double computeVignettingMult(const InputDefinition& im, const TopLeftCoords2& uv) const;

  /**
   * Frame radius for vignetting.
   */
  double inverseDemiDiagonalSquared;

  friend class PhotoCorrPreProcessor;
};

class HostPhotoTransform : public PhotoTransform {
 public:
  HostPhotoTransform(double inverseDemiDiagonalSquared, TransformPhotoParam hostPhotoParam);

  /**
   * Creates a PhotoTransform that maps the given input into the given panorama.
   * @param pano The PanoDefinition
   * @param im The InputDefinition
   */
  static HostPhotoTransform* create(const InputDefinition& im);

  /**
   * Transforms a single point on the CPU from input to panorama colorspace.
   * Equivalent to (mapPhotoInputToLinear o mapPhotoCorrectLinear o mapPhotoLinearToPano)
   * @param time Current time value.
   * @param pano Panorama definition. Must be the same as the one that was used to create the Transform.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param uv coordinates in input space, relative to the top-left of the input. They take part in the photo transform
   * through vignetting.
   * @param rgb color in input space. In [0;255].
   * @returns color in pano space
   */
  float3 mapPhotoInputToPano(int time, const PanoDefinition& pano, const InputDefinition& im, TopLeftCoords2 uv,
                             float3 rgb) const;

  /**
   * Color-corrects a sample in linear space.
   * @param colorMult color multiplier
   * @param rgb Color In [0;255].
   */
  float3 mapPhotoCorrectLinear(float3 colorMult, float3 rgb) const;

  /**
   * Same as above, but stops just before color correction is applied.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param uv coordinates in input space, relative to the top-left of the input. They take part in the photo transform
   * through vignetting.
   * @param rgb color in input space. In [0;255].
   * @returns color in pano space
   */
  virtual float3 mapPhotoInputToLinear(const InputDefinition& im, TopLeftCoords2 uv, float3 rgb) const = 0;

  /**
   * Takes a color-corrected sample in linear photo space and transform it to pano space.
   * @param rgb Color In [0;255].
   */
  virtual float3 mapPhotoLinearToPano(float3 rgb) const = 0;

  /**
   * Takes a pano space sample and transforms it to color-corrected linear photo space.
   * @param rgb Color In [0;255].
   */
  virtual float3 mapPhotoPanoToLinear(float3 rgb) const = 0;

  const TransformPhotoParam& getHostPhotoParam() const { return hostPhotoParam; }

 protected:
  TransformPhotoParam hostPhotoParam;
};

class DevicePhotoTransform : public PhotoTransform {
 public:
  DevicePhotoTransform(double inverseDemiDiagonalSquared, TransformPhotoParam devicePhotoParam);

  static DevicePhotoTransform* create(const InputDefinition& im);

  const TransformPhotoParam& getDevicePhotoParam() const { return devicePhotoParam; }

 protected:
  TransformPhotoParam devicePhotoParam;
};

}  // namespace Core
}  // namespace VideoStitch

#endif
