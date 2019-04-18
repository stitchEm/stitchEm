// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CV_IMAGE_HPP_
#define CV_IMAGE_HPP_

#include "gpu/hostBuffer.hpp"

#include <opencv2/core/core.hpp>

#include <vector>
#include <memory>

namespace VideoStitch {
namespace Calibration {

/**
 * @brief A wrapper around cv::Mat whose destructor frees the allocated memory automatically
 */
class CvImage : public cv::Mat {
 public:
  CvImage(GPU::HostBuffer<unsigned char> buf, const int w, const int h)
      : cv::Mat(cv::Size(w, h), CV_8UC1, buf.hostPtr(), cv::Mat::AUTO_STEP), hostBuf(buf) {}

  ~CvImage() { hostBuf.release(); }

 private:
  GPU::HostBuffer<unsigned char> hostBuf;
};

/* A list of images seen by the same camera */
typedef std::vector<std::shared_ptr<CvImage>> CvImages;

/* A set of cameras, seeing a list of images */
typedef std::vector<CvImages> RigCvImages;

}  // namespace Calibration
}  // namespace VideoStitch

#endif
