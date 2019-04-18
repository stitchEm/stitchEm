// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <functional>

#include "liveoutputfactory.hpp"

#include "libvideostitch-gui/utils/bitratemodeenum.hpp"
#include "libvideostitch-gui/mainwindow/stitcheroculuswindow.hpp"

#include "libvideostitch/allocator.hpp"

/**
 * @brief Meant to instantiate the widget in the GUI thread
 */
class OculusWindowFactory : public QObject {
  Q_OBJECT
 public:
  OculusWindowFactory() {}
  ~OculusWindowFactory() {}

#if defined(Q_OS_WIN)
  std::shared_ptr<StitcherOculusWindow> stitcherOculusWindow;
#endif

 public slots:
#if defined(Q_OS_WIN)
  std::shared_ptr<StitcherOculusWindow> createOculusWindow();
#endif
  void closeOculusWindow();
};

/**
 * @brief Wrapper for an oculus view window
 */
class LiveRendererOculus : public LiveRendererFactory {
  Q_OBJECT
 public:
  explicit LiveRendererOculus(const VideoStitch::OutputFormat::OutputFormatEnum type);

  virtual const QString getIdentifier() const override;

  virtual const QString getOutputDisplayName() const override { return QString(); }
  virtual bool earlyClosingRequired() const override { return true; }

  VideoStitch::Ptv::Value* serialize() const override;

  VideoStitch::PotentialValue<std::shared_ptr<VideoStitch::Core::PanoRenderer>> createRenderer() override;

  void destroyRenderer(bool wait) override;

  virtual QPixmap getIcon() const override;
  virtual QWidget* createStatusWidget(QWidget* const parent) override;

 private:
  void closeOculusWindow();

  OculusWindowFactory windowMaker;
};
