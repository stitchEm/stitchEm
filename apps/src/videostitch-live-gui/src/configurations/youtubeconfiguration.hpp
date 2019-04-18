// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef YOUTUBECONFIGURATION_HPP
#define YOUTUBECONFIGURATION_HPP

#include <array>
#include <memory>

#include <QItemSelection>

#include "istreamingserviceconfiguration.hpp"

namespace Ui {
class YoutubeConfiguration;
const int PAGE_BROADCASTS = 0;
const int PAGE_SELECT_ACCOUNT = 1;
}  // namespace Ui

class YoutubeBroadcastModel;
class GoogleCredentialModel;

class YoutubeConfiguration : public QFrame {
  Q_OBJECT

 public:
  explicit YoutubeConfiguration(QWidget *parent);
  ~YoutubeConfiguration();

 signals:
  void notifySettingsSaved(const QString &broadcastId);

 private slots:
  void handleStateChanged();
  void updateYoutubeData();
  void authenticate();
  void accountSelected(const QModelIndex &selected);
  void changeAccount();
  void onStackedWidgetPageChanged(int pageId);
  void onSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);

  void toggleAccountInfoVisible(bool visible);

 private:
  QScopedPointer<Ui::YoutubeConfiguration> ui;
  std::shared_ptr<YoutubeBroadcastModel> youtubeBroadcastModel;
  std::shared_ptr<GoogleCredentialModel> googleCredentialModel;
};

#endif  // YOUTUBECONFIGURATION_HPP
