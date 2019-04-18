// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "exposureScoringProcessor.hpp"

#include "backend/common/imageOps.hpp"

#include "gpu/score/scoring.hpp"
#include "gpu/buffer.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/uniqueBuffer.hpp"

static const double LARGE_ERROR{1000.};

namespace VideoStitch {
namespace Scoring {

ExposureScoringPostProcessor* ExposureScoringPostProcessor::create() { return new ExposureScoringPostProcessor; }

/**
 * Add up the exposure diff by channel
 */
Status ExposureScoringPostProcessor::process(GPU::Buffer<uint32_t>& devBuffer, const Core::PanoDefinition& pano,
                                             frameid_t /* frame */, GPU::Stream& stream) const {
  int64_t width = (unsigned)pano.getWidth();
  int64_t height = (unsigned)pano.getHeight();

  auto hostBuf = GPU::HostBuffer<uint32_t>::allocate(width * height, "ExposureScoring processor");
  FAIL_RETURN(hostBuf.status());

  auto getData = [&]() -> Status {
    FAIL_RETURN(GPU::memcpyAsync(hostBuf.value(), devBuffer.as_const(), stream));
    return stream.synchronize();
  };

  const Status haveData = getData();
  if (!haveData.ok()) {
    hostBuf.value().release();
    return haveData;
  }

  uint32_t* rawData = hostBuf.value().hostPtr();

  std::array<int64_t, 3> acc{{0}};

  int64_t coveredPixels{0};

  for (int64_t i = 0; i < width * height; i++) {
    uint32_t value = rawData[i];
    acc[0] += Image::RGBA::r(value);
    acc[1] += Image::RGBA::g(value);
    acc[2] += Image::RGBA::b(value);
    if (Image::RGBA::a(value)) {
      coveredPixels++;
    }
  }

  hostBuf.value().release();

  if (coveredPixels == 0) {
    rgbDiff[0] = LARGE_ERROR;
    rgbDiff[1] = LARGE_ERROR;
    rgbDiff[2] = LARGE_ERROR;
  } else {
    rgbDiff[0] = (double)acc[0] / coveredPixels;
    rgbDiff[1] = (double)acc[1] / coveredPixels;
    rgbDiff[2] = (double)acc[2] / coveredPixels;
  }

  return Status::OK();
}

}  // namespace Scoring
}  // namespace VideoStitch
