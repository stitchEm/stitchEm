// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef YOUTUBEBROADCASTMODEL_HPP
#define YOUTUBEBROADCASTMODEL_HPP

#include <QStandardItemModel>
#include <QUrl>

#include <memory>
#include <unordered_map>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#endif

#include "googleapis/client/data/jsoncpp_data.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "libvideostitch-gui/utils/imagemanager.hpp"

namespace googleapis {
namespace client {
class HttpResponse;
class HttpTransportLayerConfig;
class OAuth2Credential;
}  // namespace client
}  // namespace googleapis

namespace google_youtube_api {
class LiveBroadcast;
class LiveBroadcastListResponse;
class LiveStream;
class YouTubeService;
}  // namespace google_youtube_api

class YoutubeBroadcastModel : public QStandardItemModel {
  Q_DECLARE_TR_FUNCTIONS(YoutubeBroadcastModel)
 public:
  enum class YoutubeModelColumn { Thumbnail, Name, StartTime, PrivacyStatus };

  static bool isBroadcastStreamable(std::shared_ptr<googleapis::client::OAuth2Credential> credential,
                                    QString broadcastId, QString& errorMessage);

  explicit YoutubeBroadcastModel(std::shared_ptr<googleapis::client::OAuth2Credential> credential =
                                     std::shared_ptr<googleapis::client::OAuth2Credential>());

  bool updateUserData(const std::shared_ptr<googleapis::client::OAuth2Credential>& value,
                      const QString& broadcastId = QString(), QString* errorMessage = nullptr);

  QVariant data(const QModelIndex& index, int role) const override;
  std::shared_ptr<googleapis::client::OAuth2Credential> getAuthenticationCredential() const;
  QUrl getUrlForIndex(const QModelIndex& index, const string& resolution = "", const string& fps = "") const;
  QString getBroadcastIdForIndex(const QModelIndex& index) const;

 private:
  static bool checkAndLogResponse(googleapis::client::HttpResponse* response);
  static std::string getFormat(const std::string& resolution, const std::string& fps);
  bool initService();

  void setAuthenticationCredential(const std::shared_ptr<googleapis::client::OAuth2Credential>& value);
  bool fillUserData(const QString& broadcastId = QString(), QString* errorMessage = nullptr);
  bool checkFormatCompatible(const std::string& streamId, const std::string& resolution, const std::string& fps) const;
  void constructModel(const googleapis::client::JsonCppArray<google_youtube_api::LiveBroadcast>& broadcasts);

  QVariant broadcastImageData(const QModelIndex& index, int role) const;

  std::string checkAndUpdateStream(const std::string& streamId, const std::string& broadcastId,
                                   const std::string& resolution, const std::string& fps) const;

  std::shared_ptr<google_youtube_api::LiveStream> getStreamById(
      const std::string& streamId, const std::string& requestPart = "id,snippet,cdn,status") const;

  std::shared_ptr<google_youtube_api::LiveBroadcast> getBroadcastById(
      const std::string& broadcastId, const std::string& requestPart = "id,status,snippet,contentDetails") const;

  /**
   * @brief Retrievs list of broadcasts of current user
   *  The broadcastId and broadcastsStatus are two mutually exclusive filters.
   *  If broadcastId is present broadcastStatus would be ignored.
   *  One of the filters have to be presented
   * @param broadcastStatus Filter by status of a broadcast. Possible values are:
   *  active - Return current live broadcasts.
   *  all - Return all broadcasts.
   *  completed - Return broadcasts that have already ended.
   *  upcoming - Return broadcasts that have not yet started.
   * @param broadcastId Unique Id of broadcast to retrieve.
   * @param broadcastsListPart Specifies what parts of LiveBroadcasts object to retrieve
   * @param broadcastCount sepcifies number of broadcasts to retrieve. Accepted values is from 0 to 50.
   * @returns False if no such output exists. True if output was successfuly disabled
   */
  std::shared_ptr<google_youtube_api::LiveBroadcastListResponse> getBroadcastsList(
      const std::string& broadcastStatus = "upcoming", const string& broadcastId = "",
      const string& broadcastsListPart = "id,status,snippet,contentDetails", const int broadcastCount = 50) const;

  std::shared_ptr<google_youtube_api::LiveStream> createAndBindStream(const std::string& broadcastId,
                                                                      const std::string& resolution,
                                                                      const std::string& fps) const;
  bool unbindBroadcastStream(const std::string& broadcastId) const;

  std::unique_ptr<google_youtube_api::LiveBroadcast> bindBroadcastStream(const std::string& broadcastId,
                                                                         const std::string& streamId) const;

  std::shared_ptr<google_youtube_api::LiveStream> createNewStream(const std::string& streamName,
                                                                  const std::string& resolution,
                                                                  const std::string& fps) const;

  QUrl getUrlFromStreamId(const std::string& streamId) const;
  QUrl getUrlFromStream(const google_youtube_api::LiveStream& stream) const;

  std::shared_ptr<googleapis::client::OAuth2Credential> authenticationCredential;
  std::unique_ptr<googleapis::client::HttpTransportLayerConfig> httpTransportLayerConfig;
  std::unique_ptr<google_youtube_api::YouTubeService> youtubeService;

  static const QHash<QString, QString> privacyIconMap;

  static const QString timeFormat;

  std::unique_ptr<ImageManager> imageManager;

  bool initState = false;
};

#endif  // YOUTUBEBROADCASTMODEL_HPP
