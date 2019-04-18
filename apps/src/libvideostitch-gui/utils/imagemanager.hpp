// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef IMAGEMANAGER_HPP
#define IMAGEMANAGER_HPP

#include <memory>

#include <QNetworkAccessManager>
#include <QPixmap>
#include <QHash>
#include <QUrl>

#include "libvideostitch-gui/common.hpp"

class VS_GUI_EXPORT ImageManager : public QObject {
  Q_OBJECT

 public:
  explicit ImageManager(QObject* parent = 0);

  QPixmap getPixmapForUrl(const QUrl& fileUrl) const;
  void clearCache();

 signals:
  void imageDataUpdated(const QUrl& imageUrl);

 private:
  static const QString STANDARD_THUMBNAIL;

  void downloadFinished(QNetworkReply* reply);

  std::unique_ptr<QNetworkAccessManager> networkAccessManager;
  QHash<QUrl, QPixmap> imageCache;
};

#endif  // IMAGEMANAGER_HPP
