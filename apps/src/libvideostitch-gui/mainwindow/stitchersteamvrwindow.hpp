// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QtGlobal>

#include "libvideostitch-base/steamvrwindow.hpp"
#include "libvideostitch-base/texture.hpp"
#include "libvideostitch/allocator.hpp"

class VS_GUI_EXPORT StitcherSteamVRWindow : public SteamVRWindow, public VideoStitch::Core::PanoRenderer {
  Q_OBJECT
 public:
  const static std::string name;

  explicit StitcherSteamVRWindow(bool isStereo);
  virtual ~StitcherSteamVRWindow();

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
