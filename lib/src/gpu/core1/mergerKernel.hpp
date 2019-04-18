// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core1/textureTarget.hpp"

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/status.hpp"

#include <stdint.h>

namespace VideoStitch {
namespace Core {
class ImageMapping;
class PanoDefinition;

/**
 * Count the number of inputs that contribute to each pano pixel
 */
Status countInputs(TextureTarget, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut,
                   const Core::ImageMapping& fromIm, GPU::Stream stream);

/**
 * Count the number of inputs that contribute to each pano pixel
 */
Status colorMap(const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut, GPU::Stream stream);

/**
 * In the overlapping areas, compute output as single error value:
 * (r0 - r1)^2 + (g0 - g1)^2 + (b0 - b1)^2
 * 0x40000000 (unused in RGB210) is set in overlapping areas.
 */
Status stitchingError(TextureTarget, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut,
                      const ImageMapping& fromIm, GPU::Stream stream);

/**
 * In the overlapping areas, compute absolute difference per channel:
 * R: abs(r0 - r1), G: abs(g0 - g1), B: (b0 - b1)
 * 0x40000000 (unused in RGB210) is set in overlapping areas.
 */
Status exposureDiffRGB(TextureTarget, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut,
                       const ImageMapping& fromIm, GPU::Stream stream);

/**
 * Zero out value and alpha of all pixels of non-overlapping areas.
 * The overlapping areas need to have been marked by a stitching error
 * kernel earlier with 0x40000000.
 */
Status disregardNoDiffArea(const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut, GPU::Stream stream);

/**
 * Maps 3 * 256 * 256 (max stitchingError) to a displayable range
 */
Status amplitude(const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut, GPU::Stream stream);

/**
 * In the overlapping areas, use either the pixels from the first or second
 * contributing input. Alternate between the inputs in a checkerboard pattern.
 */
Status checkerMerge(TextureTarget, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut,
                    const ImageMapping& fromIm, unsigned checkerSize, GPU::Stream stream);

/**
 * A kernel that stores the first two inputs as gray scale in
 * the R and G channels of the pano
 */
Status noblend(TextureTarget, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut, const ImageMapping& fromIm,
               GPU::Stream stream);
}  // namespace Core
}  // namespace VideoStitch
