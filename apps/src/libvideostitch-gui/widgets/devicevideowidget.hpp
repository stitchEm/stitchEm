// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-base/videowidget.hpp"

#include "libvideostitch/allocator.hpp"

#include <QElapsedTimer>

#include <set>

class VS_GUI_EXPORT DeviceVideoWidget : public VideoWidget, public VideoStitch::Core::PanoRenderer {
  Q_OBJECT
 public:
  explicit DeviceVideoWidget(QWidget* parent = nullptr);

  void syncOn();
  void syncOff();

  std::string getName() const override;
  void render(std::shared_ptr<VideoStitch::Core::PanoOpenGLSurface>, mtime_t) override;
  void renderCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface>, mtime_t) override;
  void renderEquiangularCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface>, mtime_t) override;

 public slots:
  void registerRenderer(std::vector<std::shared_ptr<VideoStitch::Core::PanoRenderer>>* renderers);
  void onCloseProject();

 signals:
  void gotFrame(mtime_t);

 private:
  void _renderCubemap(Texture::Type, std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface>, mtime_t);

  bool sync = true;
  mtime_t ref_vs;  // the initial time of the sequence (playing from a seek point)
  QElapsedTimer clk;

  std::shared_ptr<DeviceVideoWidget>* thisPtr;  // intentionally leaked

  std::shared_ptr<VideoStitch::Core::PanoSurface> surface = nullptr;
  std::set<std::shared_ptr<VideoStitch::Core::PanoSurface>> releaseMe;
};
