// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "liveoutputfactory.hpp"

#include "libvideostitch-gui/utils/bitratemodeenum.hpp"

class StitcherSteamVRWindow;

/**
 * @brief Meant to instantiate the widget in the GUI thread
 */
class SteamVRWindowFactory : public QObject {
  Q_OBJECT
 public:
  SteamVRWindowFactory() : stitcherSteamVRWindow(nullptr) {}
  ~SteamVRWindowFactory() {}

  std::shared_ptr<StitcherSteamVRWindow> stitcherSteamVRWindow;

 public slots:
  std::shared_ptr<StitcherSteamVRWindow> createSteamVRWindow();
  void closeSteamVRWindow();
};

/**
 * @brief Wrapper for a SteamVR view window
 */
class LiveOutputSteamVR : public LiveRendererFactory {
  Q_OBJECT
 public:
  explicit LiveOutputSteamVR(const VideoStitch::OutputFormat::OutputFormatEnum type);

  virtual const QString getIdentifier() const override;
  virtual const QString getOutputDisplayName() const override { return QString(); }
  virtual bool earlyClosingRequired() const override { return true; }

  VideoStitch::Ptv::Value* serialize() const;

  VideoStitch::PotentialValue<std::shared_ptr<VideoStitch::Core::PanoRenderer>> createRenderer() override;

  void destroyRenderer(bool wait) override;

  virtual QPixmap getIcon() const override;
  virtual QWidget* createStatusWidget(QWidget* const parent) override;

 private:
  void closeSteamVRWindow();

  SteamVRWindowFactory windowMaker;
};
