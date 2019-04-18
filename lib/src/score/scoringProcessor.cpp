// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "scoringProcessor.hpp"

#include "gpu/score/scoring.hpp"
#include "gpu/buffer.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/uniqueBuffer.hpp"

namespace VideoStitch {
namespace Scoring {

ScoringPostProcessor* ScoringPostProcessor::create() { return new ScoringPostProcessor; }

/*
 * Buffers arriving here were merged through the noblendImageMerger
 * alpha channel is zero except on areas having two mapped input
 * in this case, the red and green channels respectively contain the grayscale of the first and second mapped inputs
 * the blue channel contains 0, 1 or 2, the number of mapped inputs (independently of the alpha channel)
 */
Status ScoringPostProcessor::process(GPU::Buffer<uint32_t>& devBuffer, const Core::PanoDefinition& pano,
                                     frameid_t /* frame */, GPU::Stream& stream) const {
  unsigned width = (unsigned)pano.getWidth();
  unsigned height = (unsigned)pano.getHeight();

  auto gpuBuf1 = GPU::uniqueBuffer<float>(width * height, "Scoring processor");
  auto gpuBuf2 = GPU::uniqueBuffer<float>(width * height, "Scoring processor");
  auto gpuBuf3 = GPU::uniqueBuffer<unsigned char>(width * height, "Scoring processor");

  FAIL_RETURN(gpuBuf1.status());
  FAIL_RETURN(gpuBuf2.status());
  FAIL_RETURN(gpuBuf3.status());

  GPU::memsetToZeroAsync(gpuBuf1.borrow(), stream);
  GPU::memsetToZeroAsync(gpuBuf2.borrow(), stream);
  GPU::memsetToZeroAsync(gpuBuf3.borrow(), stream);

  FAIL_RETURN(Image::splitNoBlendImageMergerChannel(gpuBuf1.borrow(), gpuBuf2.borrow(), gpuBuf3.borrow(), devBuffer,
                                                    width, height, stream));

  {
    std::unique_ptr<float[]> val1(new float[width * height]);
    std::unique_ptr<float[]> val2(new float[width * height]);
    std::unique_ptr<unsigned char[]> val3(new unsigned char[width * height]);

    FAIL_RETURN(GPU::memcpyAsync(val1.get(), gpuBuf1.borrow().as_const(), stream));
    FAIL_RETURN(GPU::memcpyAsync(val2.get(), gpuBuf2.borrow().as_const(), stream));
    FAIL_RETURN(GPU::memcpyAsync(val3.get(), gpuBuf3.borrow().as_const(), stream));

    FAIL_RETURN(stream.synchronize());

    double sum1 = 0.0;
    double sum2 = 0.0;

    double ccb = 0.0;
    double cc1 = 0.0;
    double cc2 = 0.0;

    unsigned long uncovered = 0;
    unsigned long count = 0;

    for (unsigned int i = 0; i < width * height; i++) {
      if (val3[i]) {
        uncovered++;
      }
      if (val1[i] < 0 || val2[i] < 0) {
        continue;
      }
      sum1 += (double)val1[i];
      sum2 += (double)val2[i];
      count++;
    }

    double mean1 = sum1 / (double)count;
    double mean2 = sum2 / (double)count;

    for (unsigned int i = 0; i < width * height; i++) {
      if (val1[i] < 0) continue;
      if (val2[i] < 0) continue;
      double c1 = ((double)(double)val1[i]) - mean1;
      double c2 = ((double)(double)val2[i]) - mean2;

      ccb += c1 * c2;
      cc1 += c1 * c1;
      cc2 += c2 * c2;
    }

    // return cross-correlation score between 0 and 1
    m_score = 0.5 * (1.0 + (ccb / (sqrt(cc1) * sqrt(cc2))));
    // return uncovered surface ratio
    m_uncovered = double(uncovered) / (width * height);
  }

  return Status::OK();
}

void ScoringPostProcessor::getScore(double& normalized_cross_correlation, double& uncovered_ratio) const {
  normalized_cross_correlation = m_score;
  uncovered_ratio = m_uncovered;
}

}  // namespace Scoring
}  // namespace VideoStitch
