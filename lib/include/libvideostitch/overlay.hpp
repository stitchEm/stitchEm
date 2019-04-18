// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "frame.hpp"

#include <vector>
#include <memory>

namespace VideoStitch {
namespace Ptv {
class Value;
}

namespace Core {
class PanoDefinition;
class PanoOpenGLSurface;
}  // namespace Core

namespace GPU {

/**
 * A overlayer class that is used to add image/logo on the stitch equirectangular frame.
 *
 * A overlayer can take one or more inputs, each input has its own position curve.
 * A OpenGL context should be created before create overlayer.
 * The attachContext() and detachContext() need to be overridden on the application side.
 */
class VS_EXPORT Overlayer {
 public:
  class Pimpl;

  /**
   * @brief Construct from a Ptv::Value and Core::PanoDefinition.
   * @param value parameters of overlay inputs
   * @param pano parameters of output
   */
  Overlayer();

  virtual ~Overlayer();

  /**
   * @brief Initialize used OpenGL components: VAO, VBO, PBO, FBO, shader, texture.
   * @param frameRate Stitcher frameRate
   */
  virtual void initialize(const Core::PanoDefinition* pano, const FrameRate& frameRate);

  /**
   * @brief Compute overlay.
   * @param surf Input PanoOpenGLSurface surface
   * @param oglSurf Output PanoOpenGLSurface surface
   * @param date High precesion date
   */
  virtual void computeOverlay(std::shared_ptr<Core::PanoOpenGLSurface> surf,
                              std::shared_ptr<Core::PanoOpenGLSurface> oglSurf, mtime_t date);

  /**
   * @brief Rend overlay result to the current OpenGL Context.
   * @param winWidth Current OpenGL window width
   * @param winHeight OCurrent OpenGL window height
   */
  virtual void renderOverlay(int winWidth, int winHeight);

  /**
   * @brief Attach the current OpenGL context to overlayer.
   */
  virtual void attachContext() = 0;

  /**
   * @brief Detach the current OpenGL context to overlayer.
   */
  virtual void detachContext() = 0;

 private:
  Pimpl* const pimpl;
};

}  // namespace GPU
}  // namespace VideoStitch
