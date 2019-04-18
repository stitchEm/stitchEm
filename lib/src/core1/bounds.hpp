// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "textureTarget.hpp"

#include "gpu/buffer.hpp"
#include "gpu/hostBuffer.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/output.hpp"
#include "libvideostitch/status.hpp"

#include <stdint.h>

namespace VideoStitch {
namespace Core {

class ImageMapping;
class StereoRigDefinition;

/**
 * Find min/max for each image in the X direction.
 * Do that in parallel for each row on the GPU, then reduce the data.
 * (min values are signed, with negative values meaning that the minimum wraps.
 *   Max cannot be negative.)
 * @param croppedWidth Panorama width.
 * @param imageMappings Vector of image mappings.
 * @param numInputs Number of mappings.
 * @param panoDevOut Panorama device buffer containing setup image.
 * @param tmpHostBuffer should be of size at least (pano.getWidth() * 4) bytes.
 * @param tmpDevBuffer should be of size at least (pano.getWidth() * 4) bytes.
 * @param stream Where to do the computations.
 */
Status computeHBounds(TextureTarget, int64_t croppedWidth, int64_t croppedHeight,
                      std::map<readerid_t, VideoStitch::Core::ImageMapping*>& imageMappings,
                      const StereoRigDefinition* rigDef, Eye eye, GPU::Buffer<const uint32_t> panoDevOut,
                      GPU::HostBuffer<uint32_t> tmpHostBuffer, GPU::Buffer<uint32_t> tmpDevBuffer, GPU::Stream stream,
                      bool canWrap);

/**
 * See computeHBounds.
 * @param croppedHeight Panorama height.
 * @param imageMappings Vector of image mappings.
 * @param numInputs Number of mappings.
 * @param panoDevOut Panorama device buffer containing setup image.
 * @param tmpDevBuffer should be of size at least (pano.getHeight() * 4) bytes.
 * @param stream Where to do the computations.
 */
Status computeVBounds(TextureTarget, int64_t croppedWidth, int64_t croppedHeight,
                      std::map<readerid_t, VideoStitch::Core::ImageMapping*>& imageMappings,
                      GPU::Buffer<const uint32_t> panoDevOut, GPU::HostBuffer<uint32_t> tmpHostBuffer,
                      GPU::Buffer<uint32_t> tmpDevBuffer, GPU::Stream stream);

void findMostNonSetPixels(const int min, const int croppedWidth, const uint32_t mask, const uint32_t* buffer,
                          int& mostNonSetPixels, int& mostNonSetPixelsStart);

bool findMinMaxSetPixels(const uint32_t* buffer, uint32_t mask, int bufSize, int* min, int* max);

}  // namespace Core
}  // namespace VideoStitch
