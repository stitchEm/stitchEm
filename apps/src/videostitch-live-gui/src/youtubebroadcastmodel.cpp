// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "youtubebroadcastmodel.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#endif

#include "google/youtube_api/you_tube_service.h"
#include "googleapis/client/auth/oauth2_authorization.h"
#include "googleapis/client/transport/curl_http_transport.h"
#include "googleapis/client/data/data_reader.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "libvideostitch/logging.hpp"

#include "guiconstants.hpp"

using googleapis::client::HttpResponse;
using googleapis::client::HttpTransportLayerConfig;
using googleapis::client::JsonCppArray;
using googleapis::client::OAuth2Credential;

using google_youtube_api::CdnSettings;
using google_youtube_api::LiveBroadcast;
using google_youtube_api::LiveBroadcastListResponse;
using google_youtube_api::LiveBroadcastsResource_ListMethod;
using google_youtube_api::LiveStream;
using google_youtube_api::LiveStreamListResponse;
using google_youtube_api::LiveStreamsResource_InsertMethod;
using google_youtube_api::LiveStreamsResource_ListMethod;
using google_youtube_api::YouTubeService;

const QString YoutubeBroadcastModel::timeFormat = "MMMM d, yyyy. hh:mm ap";
const QHash<QString, QString> YoutubeBroadcastModel::privacyIconMap = {
    {"public", ":/assets/icon/live/globe.png"},
    {"unlisted", ":/assets/icon/live/chain_link.png"},
    {"private", ":/assets/icon/live/lock.png"}};

bool YoutubeBroadcastModel::isBroadcastStreamable(std::shared_ptr<googleapis::client::OAuth2Credential> credential,
                                                  QString broadcastId, QString& errorMessage) {
  if (broadcastId.isEmpty()) {
    errorMessage = tr("Invalid broadcast");
    return false;
  }
  if (!credential) {
    errorMessage = tr("Invalid credential");
    return false;
  }

  googleapis::util::Status status;
  std::unique_ptr<googleapis::client::HttpTransportLayerConfig> transportLayerConfig(new HttpTransportLayerConfig());
  transportLayerConfig->ResetDefaultTransportFactory(
      new googleapis::client::CurlHttpTransportFactory(transportLayerConfig.get()));
  auto transport = transportLayerConfig->NewDefaultTransport(&status);
  if (!status.ok()) {
    errorMessage = tr("Error with http request: %0").arg(QString::fromStdString(status.error_message()));
    return false;
  }

  std::unique_ptr<google_youtube_api::YouTubeService> youtubeService(new YouTubeService(transport));
  std::unique_ptr<LiveBroadcastsResource_ListMethod> listBroadcasts(
      youtubeService->get_live_broadcasts().NewListMethod(credential.get(), "id,status"));
  listBroadcasts->set_id(broadcastId.toStdString());
  listBroadcasts->set_max_results(1);
  std::shared_ptr<LiveBroadcastListResponse> listBroadcastsResponse(LiveBroadcastListResponse::New());
  status = listBroadcasts->ExecuteAndParseResponse(listBroadcastsResponse.get());
  checkAndLogResponse(listBroadcasts->http_response());
  if (!status.ok()) {
    errorMessage = tr("Error while retrieving broadcast: %0").arg(QString::fromStdString(status.error_message()));
    return false;
  }
  auto items = listBroadcastsResponse->get_items();
  if (items.empty()) {
    errorMessage = tr("The event has been deleted.\nPlease reconfigure it or create a new one");
    return false;
  }

  auto broadCastStatus = items.get(0).get_status();
  auto lifeCycleStatus = broadCastStatus.get_life_cycle_status();
  if (lifeCycleStatus == "complete") {
    errorMessage = tr("The event has already been used.\nPlease reconfigure it or create a new one");
    return false;
  }

  return true;
}

YoutubeBroadcastModel::YoutubeBroadcastModel(std::shared_ptr<OAuth2Credential> credential)
    : httpTransportLayerConfig(new HttpTransportLayerConfig()), imageManager(new ImageManager()) {
  connect(imageManager.get(), &ImageManager::imageDataUpdated, this, [this]() {
    emit dataChanged(this->index(static_cast<int>(YoutubeModelColumn::Thumbnail), 0),
                     this->index(static_cast<int>(YoutubeModelColumn::Thumbnail), rowCount() - 1),
                     {Qt::DecorationRole, Qt::SizeHintRole});
  });

  authenticationCredential = credential;

  initService();

  if (credential) {
    fillUserData();
  }
}

bool YoutubeBroadcastModel::initService() {
  googleapis::util::Status status;

  auto factory = new googleapis::client::CurlHttpTransportFactory(httpTransportLayerConfig.get());
  httpTransportLayerConfig->ResetDefaultTransportFactory(factory);
  auto transport = httpTransportLayerConfig->NewDefaultTransport(&status);

  if (status.ok()) {
    youtubeService.reset(new YouTubeService(transport));
  } else {
    VideoStitch::Logger::get(VideoStitch::Logger::Error) << "Failed to initialize YouTube Service" << std::endl;
  }

  initState = status.ok();
  return status.ok();
}

bool YoutubeBroadcastModel::checkFormatCompatible(const std::string& streamId, const std::string& resolution,
                                                  const std::string& fps) const {
  auto currentStream = getStreamById(streamId);
  std::string youtubeFormat = getFormat(resolution, fps);
  if (!currentStream || currentStream->get_cdn().get_format() != youtubeFormat) {
    return false;
  }
  return true;
}

void YoutubeBroadcastModel::constructModel(
    const googleapis::client::JsonCppArray<google_youtube_api::LiveBroadcast>& broadcasts) {
  for (const auto& broadcast : broadcasts) {
    QList<QStandardItem*> newRow;

    auto imageItem = new QStandardItem();
    imageItem->setData(
        QString::fromStdString(broadcast.get_snippet().get_thumbnails().get_medium().get_url().ToString()),
        Qt::UserRole + 1);
    newRow.append(imageItem);

    auto headItem = new QStandardItem(QString::fromStdString(broadcast.get_snippet().get_title().ToString()));
    headItem->setData(QString::fromStdString(broadcast.get_id().ToString()), Qt::UserRole + 1);

    newRow.append(headItem);

    auto hTime = QDateTime::fromTime_t(broadcast.get_snippet().get_scheduled_start_time().ToEpochTime());
    newRow.append(new QStandardItem(hTime.toLocalTime().toString(timeFormat)));

    auto privacyStatusName = QString::fromStdString(broadcast.get_status().get_privacy_status().ToString());
    auto privacyStatusItem = new QStandardItem();
    auto pixmap = QPixmap(privacyIconMap.value(privacyStatusName));
    pixmap = pixmap.scaledToWidth(YOUTUBE_PRIVACY_WIDTH);
    privacyStatusItem->setData(pixmap, Qt::DecorationRole);
    newRow.append(privacyStatusItem);

    this->appendRow(newRow);
  }
}

QVariant YoutubeBroadcastModel::broadcastImageData(const QModelIndex& index, int role) const {
  auto url = index.data(Qt::UserRole + 1).toUrl();
  auto pixmap = imageManager->getPixmapForUrl(url);
  pixmap = pixmap.scaledToHeight(YOUTUBE_ICON_HEIGHT);

  if (role == Qt::DecorationRole) {
    return pixmap;
  }

  return pixmap.size();
}

std::string YoutubeBroadcastModel::checkAndUpdateStream(const std::string& streamId, const std::string& broadcastId,
                                                        const std::string& resolution, const std::string& fps) const {
  auto checkResutl = checkFormatCompatible(streamId, resolution, fps);
  if (checkResutl) {
    return streamId;
  }

  VideoStitch::Logger::get(VideoStitch::Logger::Info)
      << "Stream settings incompatible with output - creating new stream" << std::endl;

  unbindBroadcastStream(broadcastId);
  auto stream = createAndBindStream(broadcastId, resolution, fps);
  if (!stream) {
    return std::string();
  }

  return stream->get_id().ToString();
}

std::shared_ptr<LiveStream> YoutubeBroadcastModel::getStreamById(const std::string& streamId,
                                                                 const std::string& requestPart) const {
  std::unique_ptr<LiveStreamsResource_ListMethod> listStreams(
      youtubeService->get_live_streams().NewListMethod(authenticationCredential.get(), requestPart));

  listStreams->set_id(streamId);

  std::unique_ptr<LiveStreamListResponse> streamResponce(LiveStreamListResponse::New());
  listStreams->ExecuteAndParseResponse(streamResponce.get()).IgnoreError();
  if (!checkAndLogResponse(listStreams->http_response()) || !streamResponce->has_items()) {
    std::shared_ptr<LiveStream>();
  }

  const auto& streams = streamResponce->mutable_items();

  auto result = LiveStream::New();
  result->CopyFrom(streams.get(0));
  return std::shared_ptr<LiveStream>(result);
}

std::shared_ptr<google_youtube_api::LiveBroadcast> YoutubeBroadcastModel::getBroadcastById(
    const std::string& broadcastId, const std::string& requestPart) const {
  auto broadcastsResponse = getBroadcastsList("", broadcastId, requestPart);

  if (!broadcastsResponse || broadcastsResponse->get_items().empty()) {
    return std::shared_ptr<LiveBroadcast>();
  }

  auto result = LiveBroadcast::New();
  result->CopyFrom(broadcastsResponse->get_items().get(0));
  return std::shared_ptr<LiveBroadcast>(result);
}

std::shared_ptr<LiveBroadcastListResponse> YoutubeBroadcastModel::getBroadcastsList(
    const string& broadcastStatus, const string& broadcastId, const std::string& broadcastsListPart,
    const int broadcastCount) const {
  std::unique_ptr<LiveBroadcastsResource_ListMethod> listBroadcasts(
      youtubeService->get_live_broadcasts().NewListMethod(authenticationCredential.get(), broadcastsListPart));

  if (broadcastId.empty()) {
    listBroadcasts->set_broadcast_status(broadcastStatus);
  } else {
    listBroadcasts->set_id(broadcastId);
  }

  listBroadcasts->set_max_results(broadcastCount);
  std::shared_ptr<LiveBroadcastListResponse> listBroadcastsResponce(LiveBroadcastListResponse::New());
  auto result = listBroadcasts->ExecuteAndParseResponse(listBroadcastsResponce.get()).ok();

  checkAndLogResponse(listBroadcasts->http_response());
  if (!result) {
    return std::shared_ptr<google_youtube_api::LiveBroadcastListResponse>();
  }

  return listBroadcastsResponce;
}

std::shared_ptr<LiveStream> YoutubeBroadcastModel::createAndBindStream(const std::string& broadcastId,
                                                                       const std::string& resolution,
                                                                       const std::string& fps) const {
  auto newStream = createNewStream(std::string("Stream for ") + broadcastId, resolution, fps);

  if (newStream) {
    bindBroadcastStream(broadcastId, newStream->get_id().ToString());
  }

  return newStream;
}

bool YoutubeBroadcastModel::unbindBroadcastStream(const std::string& broadcastId) const {
  auto broadcast = getBroadcastById(broadcastId, "id,contentDetails");
  if (!broadcast || !broadcast->has_content_details()) {
    return false;
  }

  const auto& details = broadcast->get_content_details();

  if (!details.has_bound_stream_id()) {
    return true;
  }

  return (bool)bindBroadcastStream(broadcastId, details.get_bound_stream_id().ToString());
}

std::unique_ptr<LiveBroadcast> YoutubeBroadcastModel::bindBroadcastStream(const std::string& broadcastId,
                                                                          const std::string& streamId) const {
  std::unique_ptr<google_youtube_api::LiveBroadcastsResource_BindMethod> bindMethod(
      youtubeService->get_live_broadcasts().NewBindMethod(authenticationCredential.get(), broadcastId,
                                                          "id,contentDetails"));
  bindMethod->set_stream_id(streamId);

  std::unique_ptr<LiveBroadcast> boundBroadcast(LiveBroadcast::New());
  boundBroadcast->mutable_contentDetails().set_projection("360");
  bindMethod->ExecuteAndParseResponse(boundBroadcast.get()).IgnoreError();
  if (!checkAndLogResponse(bindMethod->http_response())) {
    return std::unique_ptr<LiveBroadcast>();
  }
  return boundBroadcast;
}

std::shared_ptr<google_youtube_api::LiveStream> YoutubeBroadcastModel::createNewStream(const std::string& streamName,
                                                                                       const std::string& resolution,
                                                                                       const std::string& fps) const {
  std::unique_ptr<google_youtube_api::LiveStream> stream(google_youtube_api::LiveStream::New());

  auto stream_snippet = stream->mutable_snippet();
  stream_snippet.set_title(streamName);

  CdnSettings stream_cdn = stream->mutable_cdn();
  stream_cdn.set_format(getFormat(resolution, fps));
  stream_cdn.set_ingestion_type("rtmp");
  stream_cdn.set_resolution(resolution);
  stream_cdn.set_frame_rate(fps);

  string stream_parts = "snippet,cdn";
  std::unique_ptr<LiveStreamsResource_InsertMethod> insert_stream(
      youtubeService->get_live_streams().NewInsertMethod(authenticationCredential.get(), stream_parts, *stream));

  std::shared_ptr<LiveStream> got_stream(LiveStream::New());
  insert_stream->ExecuteAndParseResponse(got_stream.get()).IgnoreError();

  if (!checkAndLogResponse(insert_stream->http_response())) {
    return std::shared_ptr<LiveStream>();
  }

  return got_stream;
}

QUrl YoutubeBroadcastModel::getUrlFromStreamId(const std::string& streamId) const {
  auto streamPtr = getStreamById(streamId);
  if (streamPtr) {
    return getUrlFromStream(*streamPtr);
  }
  return QUrl();
}

QUrl YoutubeBroadcastModel::getUrlFromStream(const LiveStream& stream) const {
  auto streamCdnIngestionInfo = stream.get_cdn().get_ingestion_info();
  auto ingestionAdress = streamCdnIngestionInfo.get_ingestion_address();
  auto streamName = streamCdnIngestionInfo.get_stream_name();

  return QUrl(QString::fromStdString(ingestionAdress.ToString() + "/" + streamName.ToString()));
}

bool YoutubeBroadcastModel::fillUserData(const QString& broadcastId, QString* errorMessage) {
  if (!authenticationCredential || !initState) {
    if (errorMessage) {
      *errorMessage = tr("YouTube feature was not initialized properly");
    }
    VideoStitch::Logger::get(VideoStitch::Logger::Error) << "YouTube model was not initialized properly" << std::endl;
    return false;
  }

  auto broadcastsResponse = getBroadcastsList("upcoming", broadcastId.toStdString());
  if (broadcastsResponse) {
    auto items = broadcastsResponse->get_items();

    // When retrieving data for a specific event
    if (!broadcastId.isEmpty()) {
      if (items.empty()) {
        if (errorMessage) {
          *errorMessage = tr("The event has been deleted.\nPlease reconfigure it or create a new one");
        }
        return false;
      }

      auto broadCastStatus = items.get(0).get_status();
      auto lifeCycleStatus = broadCastStatus.get_life_cycle_status();
      if (lifeCycleStatus == "complete") {
        if (errorMessage) {
          *errorMessage = tr("The event has already been used.\nPlease reconfigure it or create a new one");
        }
        return false;
      }
    }

    constructModel(items);
  }

  return true;
}

bool YoutubeBroadcastModel::updateUserData(const std::shared_ptr<googleapis::client::OAuth2Credential>& value,
                                           const QString& broadcastId, QString* errorMessage) {
  beginResetModel();
  setAuthenticationCredential(value);

  this->clear();
  imageManager->clearCache();
  auto result = fillUserData(broadcastId, errorMessage);
  endResetModel();
  return result;
}

QUrl YoutubeBroadcastModel::getUrlForIndex(const QModelIndex& index, const std::string& resolution,
                                           const std::string& fps) const {
  if (!authenticationCredential || !initState) {
    VideoStitch::Logger::get(VideoStitch::Logger::Error) << "YouTube model was not initialized properly" << std::endl;
    return QUrl();
  }

  auto broadcastId = getBroadcastIdForIndex(index).toStdString();

  std::string broadcastsListPart = "id,contentDetails";
  auto broadcast = getBroadcastById(broadcastId, broadcastsListPart);

  if (!broadcast || !broadcast->has_content_details()) {
    return QUrl();
  }
  const auto& details = broadcast->get_content_details();

  if (!details.has_bound_stream_id()) {
    auto newStream = createAndBindStream(broadcastId, resolution, fps);
    return newStream ? getUrlFromStream(*newStream) : QUrl();
  }

  auto streamId = checkAndUpdateStream(details.get_bound_stream_id().ToString(), broadcastId, resolution, fps);

  return getUrlFromStreamId(streamId);
}

QString YoutubeBroadcastModel::getBroadcastIdForIndex(const QModelIndex& index) const {
  return this->index(index.row(), static_cast<int>(YoutubeModelColumn::Name)).data(Qt::UserRole + 1).toString();
}

std::string YoutubeBroadcastModel::getFormat(const string& resolution, const string& fps) {
  std::string youtubeFormat = resolution;
  if (fps == "60fps") {
    youtubeFormat += "_hfr";
  }
  return youtubeFormat;
}

QVariant YoutubeBroadcastModel::data(const QModelIndex& index, int role) const {
  if (role == Qt::DecorationRole || role == Qt::SizeHintRole) {
    auto column = static_cast<YoutubeModelColumn>(index.column());

    switch (column) {
      case YoutubeModelColumn::Thumbnail:
        return broadcastImageData(index, role);
      default:
        break;
    }
  }

  return QStandardItemModel::data(index, role);
}

std::shared_ptr<googleapis::client::OAuth2Credential> YoutubeBroadcastModel::getAuthenticationCredential() const {
  return authenticationCredential;
}

void YoutubeBroadcastModel::setAuthenticationCredential(
    const std::shared_ptr<googleapis::client::OAuth2Credential>& value) {
  authenticationCredential = value;
}

bool YoutubeBroadcastModel::checkAndLogResponse(HttpResponse* response) {
  auto errorLog = VideoStitch::Logger::get(VideoStitch::Logger::Error);
  auto verboseLog = VideoStitch::Logger::get(VideoStitch::Logger::Verbose);
  auto debugLog = VideoStitch::Logger::get(VideoStitch::Logger::Debug);

  googleapis::util::Status transport_status = response->transport_status();

  // Rewind the stream before we dump it since this could get called after
  // ExecuteAndParseResponse which will have read the result.
  response->body_reader()->SetOffset(0);
  bool response_was_ok;
  if (!transport_status.ok()) {
    errorLog << "ERROR: " << transport_status.error_message() << std::endl;
    return false;
  } else if (!response->ok()) {
    string body;
    googleapis::util::Status status = response->GetBodyString(&body);
    if (!status.ok()) {
      body.append("ERROR reading HTTP response body: ");
      body.append(status.error_message());
    }
    errorLog << "ERROR(" << response->http_code() << "): " << body << std::endl;
    response_was_ok = false;
  } else {
    verboseLog << "OK(" << response->http_code() << ")" << std::endl;
    string body;
    googleapis::util::Status status = response->GetBodyString(&body);
    if (!status.ok()) {
      body.append("ERROR reading HTTP response body: ");
      body.append(status.error_message());
    }
    verboseLog << "----------  [begin response body]  ----------" << std::endl;
    verboseLog << body << std::endl;
    verboseLog << "-----------  [end response body]  -----------" << std::endl;
    response_was_ok = true;
  }

  // Restore offset in case someone downstream wants to read the body again.
  if (response->body_reader()->SetOffset(0) != 0) {
    debugLog << "Could not reset body offset so future reads (if any) will fail." << std::endl;
  }
  return response_was_ok;
}
