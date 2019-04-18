// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <gpu/buffer.hpp>
#include "gpu/surface.hpp"
#include <gpu/stream.hpp>

#include <inttypes.h>

namespace VideoStitch {
namespace Render {

/**
 * @brief A set of geometric primitives.
 */

/**
 * Draw a line onto a buffer
 * @param dst Destination buffer
 * @param width Destination buffer width.
 * @param height Destination buffer height.
 * @param aX Start point X.
 * @param aX Start point Y.
 * @param bX End point X.
 * @param bX End point Y.
 * @param t Line thickness.
 * @param color Line color.
 * @param stream GPU Stream to run in.
 */
template <typename Image>
Status drawLine(Image& dst, int64_t width, int64_t height, float aX, float aY, float bX, float bY, float t,
                uint32_t color, GPU::Stream stream);

/**
 * Draw a disk onto a buffer.
 * @param dst Destination buffer
 * @param width Destination buffer width.
 * @param height Destination buffer height.
 * @param aX Center X.
 * @param aX Center Y.
 * @param t Square radius.
 * @param color Line color.
 * @param stream GPU Stream to run in.
 */
template <typename Image>
Status drawDisk(Image& dst, int64_t width, int64_t height, float aX, float aY, float t, uint32_t color,
                GPU::Stream stream);

/**
 * A kernel that overlays a circle.
 * @param dst Destination buffer
 * @param width Destination buffer width.
 * @param height Destination buffer height.
 * @param centerX Circle center position on the x axis.
 * @param centerY Circle center position on the y axis.
 * @param sqrRadius Circle radius, squared.
 * @param t Circle thickness.
 * @param color Circle color.
 * @param stream GPU Stream to run in.
 */
template <typename Image>
Status drawCircle(Image& dst, int64_t width, int64_t height, float centerX, float centerY, float innerSqrRadius,
                  float outerSqrRadius, uint32_t color, GPU::Stream stream);

/**
 * A kernel that overlays a circle, top only
 * @param dst Destination buffer
 * @param width Destination buffer width.
 * @param height Destination buffer height.
 * @param centerX Circle center position on the x axis.
 * @param centerY Circle center position on the y axis.
 * @param sqrRadius Circle radius, squared.
 * @param t Circle thickness.
 * @param color Circle color.
 * @param stream GPU Stream to run in.
 */
template <typename Image>
Status drawCircleTop(Image& dst, int64_t width, int64_t height, float centerX, float centerY, float innerSqrRadius,
                     float outerSqrRadius, uint32_t color, GPU::Stream stream);

/**
 * A kernel that overlays a circle, bottom only
 * @param dst Destination buffer
 * @param width Destination buffer width.
 * @param height Destination buffer height.
 * @param centerX Circle center position on the x axis.
 * @param centerY Circle center position on the y axis.
 * @param sqrRadius Circle radius, squared.
 * @param t Circle thickness.
 * @param color Circle color.
 * @param stream GPU Stream to run in.
 */
template <typename Image>
Status drawCircleBottom(Image& dst, int64_t width, int64_t height, float centerX, float centerY, float innerSqrRadius,
                        float outerSqrRadius, uint32_t color, GPU::Stream stream);

/**
 * A kernel that overlays a circle, top right quarter only
 * @param dst Destination buffer
 * @param width Destination buffer width.
 * @param height Destination buffer height.
 * @param centerX Circle center position on the x axis.
 * @param centerY Circle center position on the y axis.
 * @param sqrRadius Circle radius, squared.
 * @param t Circle thickness.
 * @param color Circle color.
 * @param stream GPU Stream to run in.
 */
template <typename Image>
Status drawCircleTopRight(Image& dst, int64_t width, int64_t height, float centerX, float centerY, float innerSqrRadius,
                          float outerSqrRadius, uint32_t color, GPU::Stream stream);

/**
 * A kernel that overlays a circle, top right quarter only
 * @param dst Destination buffer
 * @param width Destination buffer width.
 * @param height Destination buffer height.
 * @param centerX Circle center position on the x axis.
 * @param centerY Circle center position on the y axis.
 * @param sqrRadius Circle radius, squared.
 * @param t Circle thickness.
 * @param color Circle color.
 * @param stream GPU Stream to run in.
 */
template <typename Image>
Status drawCircleBottomRight(Image& dst, int64_t width, int64_t height, float centerX, float centerY,
                             float innerSqrRadius, float outerSqrRadius, uint32_t color, GPU::Stream stream);

}  // namespace Render
}  // namespace VideoStitch
