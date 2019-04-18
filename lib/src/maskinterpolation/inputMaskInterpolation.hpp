// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "gpu/uniqueBuffer.hpp"
#include "gpu/vectorTypes.hpp"
#include "gpu/stream.hpp"
#include "core1/imageMapping.hpp"

#include "libvideostitch/status.hpp"

#include <vector>
#include <opencv2/core/core.hpp>

namespace VideoStitch {
namespace MaskInterpolation {

/*
 * This class takes two input mask s(stored in the input space) and interpolates the in-between masks.
 */

class InputMaskInterpolation {
 public:
  static Potential<InputMaskInterpolation> create(const Core::PanoDefinition& pano,
                                                  const std::map<readerid_t, Input::VideoReader*>& readers,
                                                  const int polygonSampleCount = 2000);

  ~InputMaskInterpolation();

  /**
   * @brief Setup the temporary memory for the mask interpolation
   */
  Status setup(const Core::PanoDefinition& pano, const std::map<readerid_t, Input::VideoReader*>& readers);

  /**
   * @brief Get the current start and end frame ids.
   */
  std::pair<frameid_t, frameid_t> getFrameIds() const;

  /**
   * @brief Pass the first and last frame of the mask to be interpolated
   * polygon0s, and polygon1s store the polylines of the masks in the input spaces
   */
  Status setupKeyframes(const Core::PanoDefinition& pano, const frameid_t frameId0,
                        const std::map<videoreaderid_t, std::vector<cv::Point>>& polygon0s, const frameid_t frameId1,
                        const std::map<videoreaderid_t, std::vector<cv::Point>>& polygon1s);

  /**
   * @brief Get the interpolated mask at "frame" between "frameId0" and "frameId0"
   * @inputsMap Stores the output camera index (compatible for use in the panoStitcher.cpp) in the output space
   */
  Status getInputsMap(const Core::PanoDefinition& pano, const frameid_t frame, GPU::Buffer<uint32_t> inputsMap) const;

  void deactivate();

  bool isActive();

 private:
  explicit InputMaskInterpolation(const int polygonSampleCount);

  /**
   * @brief Get the interpolated mask at "frame" between "frameId0" and "frameId0"
   * @inputs Storess the interpolated masks as polylines in the input space.
   */
  Status getInputs(const frameid_t frame, std::map<videoreaderid_t, std::vector<cv::Point>>& inputs) const;

  frameid_t frameId0, frameId1;  // These bounded frame id
  const int polygonSampleCount;  // The number of samples used to sample to polylines
  std::map<videoreaderid_t, std::vector<cv::Point2f>> sampledPoint0s;  // The sample points of the first frame
  std::map<videoreaderid_t, std::vector<cv::Point2f>> sampledPoint1s;  // The sample points of the last frame
  std::map<videoreaderid_t, std::vector<int>>
      matchIndices;  // The correspondence index of sampled points from the first frame in the final frame
  std::map<videoreaderid_t, Core::Transform*> transforms;

  mutable Core::SourceSurface* devCoord = nullptr;     // Temporary data used for interpolating the inbetween mask
  mutable GPU::UniqueBuffer<unsigned char> inputMask;  // Temporary data used for interpolating the inbetween mask
  mutable std::vector<unsigned char> masks;            // Temporary data used for interpolating the inbetween mask

  bool activated;  // Check whether interpolation is activated or not
};

}  // namespace MaskInterpolation
}  // namespace VideoStitch
