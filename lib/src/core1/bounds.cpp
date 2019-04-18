// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "bounds.hpp"

#include "gpu/memcpy.hpp"
#include "gpu/core1/boundsKernel.hpp"
#include "core1/imageMapping.hpp"

#include "libvideostitch/stereoRigDef.hpp"

#include <algorithm>

namespace VideoStitch {
namespace Core {

/**
 * Find the min and max set pixels of a buffer. Returns false if there are no set pixels.
 */
bool findMinMaxSetPixels(const uint32_t* buffer, uint32_t mask, int bufSize, int* min, int* max) {
  *min = -1;
  *max = -1;
  int i;
  for (i = 0; i < bufSize; ++i) {
    if (buffer[i] & mask) {
      *min = i;
      break;
    }
  }
  if (*min == -1) {
    return false;
  }
  for (; i < bufSize; ++i) {
    if (buffer[i] & mask) {
      *max = i;
    }
  }
  return true;
}

void findMostNonSetPixels(const int min, const int croppedWidth, const uint32_t mask, const uint32_t* buffer,
                          int& mostNonSetPixels, int& mostNonSetPixelsStart) {
  // Here it's a bit more complicated since the panorama wraps. The problem can be reduced down to the following graph
  // problem:
  //  - Each contiguous set of set pixels is represented by a red vertex (contiguity is computed in the wrapping domain,
  //  i.e. the first and last pixel are contiguous).
  //  - Each contiguous set of non-set pixels is represented by a black vertex.
  //  - To vertices are connected if the two correspond sets touch each other.
  //  - Each vertex is attributed a integer weight equal to the number of pixels in its set.
  // Obviously, this graph is either a single vertex, or a cycle of alternating black and red vertices (even number of
  // vertices). In addition, the sum of all vertex weights is the width of the pano.
  //  - If there are at least two vertices, finding the best interval is the same as finding the black vertex that can
  //  be removed such that it minimizes the sum of the weights of all other vertices.
  //    Since the sum of weights is constant, this is actually the same as finding the black vertex with the largest
  //    weight, or finding the largest contiguous (wrapping) set of non-set pixels.
  //  - If there is only one vertex, then the interval is either empty (black) or the whole thing (red).

  // We have at least one set pixel, else we would already have returned.
  // Because the problem cycles, there is a translation symmetry, so we can choose to have the first pixel be the first
  // set pixel.
#define COL(i) ((min + (i)) % (int)croppedWidth)
  assert(buffer[COL(0)] & mask);
  // Find the largest contiguous set of non-set pixels.
  mostNonSetPixels = 0;
  mostNonSetPixelsStart = -1;
  for (int i = 0; i < croppedWidth;) {
    // Skip all set pixels.
    while (i < croppedWidth && (buffer[COL(i)] & mask)) {
      ++i;
    }
    // Count all non-set pixels.
    int nonSetPixels = 0;
    const int start = COL(i);
    while (i < croppedWidth && !(buffer[COL(i)] & mask)) {
      ++nonSetPixels;
      ++i;
    }
    if (nonSetPixels > mostNonSetPixels) {
      mostNonSetPixels = nonSetPixels;
      mostNonSetPixelsStart = start;
    }
  }
#undef COL
}

Status computeHBounds(TextureTarget t, int64_t croppedWidth, int64_t croppedHeight,
                      std::map<readerid_t, VideoStitch::Core::ImageMapping*>& imageMappings,
                      const StereoRigDefinition* rigDef, Eye eye, GPU::Buffer<const uint32_t> panoDevOut,
                      GPU::HostBuffer<uint32_t> tmpHostBuffer, GPU::Buffer<uint32_t> tmpDevBuffer, GPU::Stream stream,
                      bool canWrap) {
  FAIL_RETURN(vertOr(croppedWidth, croppedHeight, panoDevOut, tmpDevBuffer, stream));

  GPU::HostBuffer<uint32_t> rowBuffer = tmpHostBuffer;
  FAIL_CAUSE(GPU::memcpyAsync(rowBuffer, tmpDevBuffer.as_const(), size_t(croppedWidth * 4), stream), Origin::Stitcher,
             ErrType::SetupFailure, "Could not compute horizontal bounds");
  FAIL_CAUSE(stream.synchronize(), Origin::Stitcher, ErrType::SetupFailure, "Could not compute horizontal bounds");

  // find min/max for each image
  for (auto mapping : imageMappings) {
    int min = -1, max = -1;
    uint32_t mask = 1 << mapping.first;
    // The problem here is to find an interval of minimum size that covers all set pixels for an image.
    // First find the first and the last set pixels.
    if (!findMinMaxSetPixels(rowBuffer, mask, (int)croppedWidth, &min, &max)) {
      // This can happen if there are no image pixels in the panorama (e.g. the image is behind us for rectilinear
      // panos). In this case we don't set bounds, and the mapper will remain empty.
      continue;
    }
    if (rigDef && rigDef->getGeometry() == StereoRigDefinition::Polygonal) {
      std::vector<int> inputs = (eye == LeftEye ? rigDef->getLeftInputs() : rigDef->getRightInputs());
      if (std::find(inputs.begin(), inputs.end(), (int)mapping.first) == inputs.end()) {
        continue;
      }
    }

    if (!canWrap || t != EQUIRECTANGULAR) {
      // Here it's simple: the lower bound is the first set pixel and the upper bound is the last set pixel.
      // Note that even if the image does not wrap through the antipode, it is still possible that there are two
      // separate continuous sets of pixels.
      mapping.second->setHBounds(t, min, max, croppedWidth);
    } else {
      int mostNonSetPixels = 0;
      int mostNonSetPixelsStart = -1;
      findMostNonSetPixels(min, int(croppedWidth), mask, rowBuffer, mostNonSetPixels, mostNonSetPixelsStart);
      if (mostNonSetPixels == 0) {
        // All set: full image.
        mapping.second->setHBounds(t, 0, (int)croppedWidth - 1, croppedWidth);
      } else {
        mapping.second->setHBounds(t, (mostNonSetPixelsStart + mostNonSetPixels) % (int)croppedWidth,
                                   mostNonSetPixelsStart - 1, croppedWidth);
      }
    }
  }
  return Status::OK();
}

Status computeVBounds(TextureTarget t, int64_t croppedWidth, int64_t croppedHeight,
                      std::map<readerid_t, VideoStitch::Core::ImageMapping*>& imageMappings,
                      GPU::Buffer<const uint32_t> panoDevOut, GPU::HostBuffer<uint32_t> tmpHostBuffer,
                      GPU::Buffer<uint32_t> tmpDevBuffer, GPU::Stream stream) {
  FAIL_RETURN(horizOr(croppedWidth, croppedHeight, panoDevOut, tmpDevBuffer, stream));

  GPU::HostBuffer<uint32_t> colBuffer = tmpHostBuffer;
  FAIL_CAUSE(GPU::memcpyAsync(colBuffer, tmpDevBuffer.as_const(), size_t(croppedHeight * 4), stream), Origin::Stitcher,
             ErrType::SetupFailure, "Could not compute vertical bounds")
  FAIL_CAUSE(stream.synchronize(), Origin::Stitcher, ErrType::SetupFailure, "Could not compute vertical bounds");

  // find min/max for each image
  for (auto mapping : imageMappings) {
    if (mapping.second->getOutputRect(t).horizontallyEmpty()) {  // skip images that don't contribute pixels.
      continue;
    }
    uint32_t mask = 1 << mapping.first;
    int min = -1, max = -1;
    // The problem here is to find an interval of minimum size that covers all set pixels for an image.
    // First find the first and the last set pixels.
    if (!findMinMaxSetPixels(colBuffer, mask, (int)croppedHeight, &min, &max)) {
      return {
          Origin::Stitcher, ErrType::SetupFailure,
          "Could not compute vertical bounds. No pixels should have been caught by the horizontal bounds computation."};
    }
    assert(min < croppedHeight);
    assert(max < croppedHeight);
    mapping.second->setVBounds(t, min, max);
  }
  return Status::OK();
}

}  // namespace Core
}  // namespace VideoStitch
