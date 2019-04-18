// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef YOUTUBEBASECONFIGURATION_HPP
#define YOUTUBEBASECONFIGURATION_HPP

#include <memory>
#include <array>
#include <QWidget>

#include "istreamingserviceconfiguration.hpp"

namespace Ui {
class YoutubeBaseConfiguration;
}

class YoutubeBroadcastModel;
class GoogleCredentialModel;

class YoutubeBaseConfiguration : public IStreamingServiceConfiguration {
  Q_OBJECT

 public:
  explicit YoutubeBaseConfiguration(QWidget* parent, LiveOutputRTMP* outputRef,
                                    LiveProjectDefinition* projectDefinition);
  ~YoutubeBaseConfiguration();

  virtual bool loadConfiguration() override;
  virtual void saveConfiguration() override;
  virtual bool hasValidConfiguration() const override;
  virtual void startBaseConfiguration() override;

 private slots:
  void updateYoutubeData(const QString& broadcastId);
  void openMoreConfiguration();
  void handleStateChanged();

 private:
  std::pair<std::string, std::string> currentYoutubeProfile() const;

  static const std::array<int, 6> widthArray;
  static const std::array<int, 6> heightArray;
  static const int HFR_THRESHOLD;
  static const int FPS_THRESHOLD;

  QScopedPointer<Ui::YoutubeBaseConfiguration> ui;
  std::shared_ptr<YoutubeBroadcastModel> youtubeBroadcastModel;
  std::shared_ptr<GoogleCredentialModel> googleCredentialModel;
};

#endif  // YOUTUBEBASECONFIGURATION_HPP
