// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QFile>
#include <QStandardPaths>
#include <QNetworkRequest>
#include <QNetworkReply>

#include "libvideostitch/logging.hpp"

#include "imagemanager.hpp"

const QString ImageManager::STANDARD_THUMBNAIL = ":/live/icons/assets/images/mqdefault_live.jpg";

QPixmap ImageManager::getPixmapForUrl(const QUrl& fileUrl) const {
  if (imageCache.contains(fileUrl)) {
    return imageCache[fileUrl];
  }

  networkAccessManager->get(QNetworkRequest(fileUrl));

  return QPixmap(STANDARD_THUMBNAIL);
}

void ImageManager::clearCache() { imageCache.clear(); }

ImageManager::ImageManager(QObject* parent) : QObject(parent), networkAccessManager(new QNetworkAccessManager()) {
  connect(networkAccessManager.get(), &QNetworkAccessManager::finished, this, &ImageManager::downloadFinished);
}

void ImageManager::downloadFinished(QNetworkReply* reply) {
  QUrl url = reply->url();
  if (reply->error()) {
    VideoStitch::Logger::get(VideoStitch::Logger::Debug)
        << "Download of " << url.toEncoded().constData() << " failed: " << reply->errorString().toStdString()
        << std::endl;
    return;
  }

  VideoStitch::Logger::get(VideoStitch::Logger::Debug)
      << "Download of " << url.toEncoded().constData() << " succeded" << std::endl;

  QPixmap result;
  result.loadFromData(reply->readAll());
  imageCache[url] = result;
  emit imageDataUpdated(url);
}
