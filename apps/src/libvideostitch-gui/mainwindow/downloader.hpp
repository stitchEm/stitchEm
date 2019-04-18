// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QObject>

class QNetworkAccessManager;
class QNetworkReply;
class QTemporaryFile;
class QUrl;

namespace VideoStitch {
namespace Helper {
class VS_GUI_EXPORT Downloader : public QObject {
  Q_OBJECT

 public:
  explicit Downloader(QObject* parent = nullptr);
  ~Downloader();
  bool hasBeenAborted() const;

 public slots:
  QString startDownload(const QUrl& url, const QString& outputFileTemplate);

  void abort();

 signals:
  void progression(int percent);
  void progressionMessage(const QString& message);

 private slots:
  void replyFinished(QNetworkReply* tempReply);
  void downloadProgress(qint64 bytesReceived, qint64 bytesTotal);
  void downloadReadyRead();

 private:
  QNetworkAccessManager* networkManager;
  QNetworkReply* currentReply;
  QScopedPointer<QTemporaryFile> outputFile;
  bool hasAborted;
};

}  // namespace Helper
}  // namespace VideoStitch
