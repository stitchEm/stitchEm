// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "autoCropAlgorithm.hpp"

#include "autoCrop.hpp"

#include "gpu/memcpy.hpp"
#include "common/container.hpp"
#include "core/controllerInputFrames.hpp"
#include "util/registeredAlgo.hpp"

#include "libvideostitch/inputFactory.hpp"

#include <opencv2/imgproc.hpp>

//#define DEBUG_AUTOCROP_ALGORITHM

namespace VideoStitch {
namespace AutoCrop {

namespace {
Util::RegisteredAlgo<AutoCropAlgorithm> registered("autocrop");
}

const char* AutoCropAlgorithm::docString =
    "An algorithm that crops images captured using circular fisheye camera automatically.\n";

AutoCropAlgorithm::AutoCropAlgorithm(const Ptv::Value* config) : autoCropConfig(config) {}

AutoCropAlgorithm::~AutoCropAlgorithm() {}

Potential<Ptv::Value> AutoCropAlgorithm::apply(Core::PanoDefinition* pano, ProgressReporter*, Util::OpaquePtr**) const {
  AutoCrop autoCrop(autoCropConfig);
  auto container = Core::ControllerInputFrames<PixelFormat::RGBA, uint32_t>::create(pano);
  FAIL_RETURN(container.status());
  std::map<readerid_t, PotentialValue<GPU::HostBuffer<uint32_t>>> loadedFrames;
  FAIL_RETURN(container->seek((int)0));
  container->load(loadedFrames);
  for (auto& it : loadedFrames) {
    readerid_t inputid = it.first;
    if (inputid > (int)pano->numInputs()) {
      continue;
    }
    auto potLoadedFrame = it.second;
    FAIL_RETURN(potLoadedFrame.status());

    GPU::HostBuffer<uint32_t> frame = potLoadedFrame.value();

    /* Get the size of the current image */
    Core::InputDefinition& inputDef = pano->getInput(inputid);
    const int width = (int)inputDef.getWidth();
    const int height = (int)inputDef.getHeight();
    auto potHostFrame = GPU::HostBuffer<unsigned char>::allocate(frame.numElements() * 4, "Autocrop frame loading");
    FAIL_RETURN(potHostFrame.status());
    GPU::HostBuffer<unsigned char> hostFrame = potHostFrame.value();
    std::memcpy(hostFrame.hostPtr(), frame.hostPtr(), frame.byteSize());
    cv::Mat cvImage;
    cv::Mat originalImage(cv::Size((int)width, (int)height), CV_8UC4, frame.hostPtr(), cv::Mat::AUTO_STEP);
    cv::cvtColor(originalImage, cvImage, CV_RGBA2BGR);

    cv::Point3i circle;
    FAIL_RETURN(autoCrop.findCropCircle(cvImage, circle));
    const cv::Point center = cv::Point(circle.x, circle.y);
    const int radius = circle.z;
    inputDef.setCropLeft(center.x - radius);
    inputDef.setCropRight(center.x + radius);
    inputDef.setCropTop(center.y - radius);
    inputDef.setCropBottom(center.y + radius);
    hostFrame.release();
    if (autoCropConfig.dumpCircleImage()) {
      FAIL_RETURN(autoCrop.dumpCircleFile(circle, inputDef.getReaderConfig().asString()));
    }
    if (autoCropConfig.dumpOriginalImage()) {
      FAIL_RETURN(autoCrop.dumpOriginalFile(inputDef.getReaderConfig().asString()));
    }
#ifdef DEBUG_AUTOCROP_ALGORITHM
    cv::imwrite("dumpImg.png", cvImage);
#endif
  }
  return Potential<Ptv::Value>(Status::OK());
}

}  // namespace AutoCrop
}  // namespace VideoStitch
