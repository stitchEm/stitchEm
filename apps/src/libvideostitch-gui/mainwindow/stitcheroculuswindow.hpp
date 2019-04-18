// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QtGlobal>

#if defined(Q_OS_WIN)

#include "libvideostitch-base/oculuswindow.hpp"

#include "libvideostitch/allocator.hpp"

class VS_GUI_EXPORT StitcherOculusWindow : public OculusWindow, public VideoStitch::Core::PanoRenderer {
  Q_OBJECT
 public:
  const static std::string name;

  explicit StitcherOculusWindow(bool isStereo, bool mirrorModeEnabled = false);
  virtual ~StitcherOculusWindow();

  /*
   * @brief Checks that oculus can be initialized
   *
   * returns true if oculus can be initialized
   * returns false otherwise
   */
  static bool checkOculusCanBeInitialized();

  std::string getName() const override;
  void render(std::shared_ptr<VideoStitch::Core::PanoOpenGLSurface>, mtime_t) override;
  void renderCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface>, mtime_t) override;
  void renderEquiangularCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface>, mtime_t) override;

 signals:
  void orientationChanged(double yaw, double pitch, double roll);

 private:
  void _renderCubemap(Texture::Type, std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface>, mtime_t);

  std::shared_ptr<VideoStitch::Core::PanoSurface> surface = nullptr;
};

#endif
