// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "downloader.hpp"

#include "libvideostitch-base/logmanager.hpp"

#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QTemporaryFile>

namespace VideoStitch {
namespace Helper {

Downloader::Downloader(QObject* parent)
    : QObject(parent), networkManager(new QNetworkAccessManager(this)), currentReply(nullptr), hasAborted(false) {}

Downloader::~Downloader() {}

bool Downloader::hasBeenAborted() const { return hasAborted; }

QString Downloader::startDownload(const QUrl& url, const QString& outputFileTemplate) {
  outputFile.reset(new QTemporaryFile(outputFileTemplate));
  outputFile->setAutoRemove(false);
  outputFile->open();  // open to have the final file name
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
      QString("Start downloading file %0").arg(outputFile->fileName()));
  QObject::connect(networkManager, &QNetworkAccessManager::finished, this, &Downloader::replyFinished);
  currentReply = networkManager->get(QNetworkRequest(url));
  return outputFile->fileName();
}

void Downloader::abort() {
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile("Abort download");
  if (!hasAborted) {
    hasAborted = true;
    currentReply->disconnect(this);
    currentReply->abort();
  }
}

void Downloader::downloadProgress(qint64 bytesReceived, qint64 bytesTotal) {
  if (!hasAborted) {
    emit progression((double)bytesReceived * 100.0 / (double)bytesTotal);
    const double divider = 1 << 20;
    emit progressionMessage(tr("Downloading %0 MB/%1 MB...")
                                .arg(QString::number((double)bytesReceived / divider, 'f', 1),
                                     QString::number((double)bytesTotal / divider, 'f', 1)));
  }
}

void Downloader::downloadReadyRead() {
  if (!hasAborted) {
    outputFile->write(currentReply->readAll());
  }
}

void Downloader::replyFinished(QNetworkReply* tempReply) {
  int httpstatuscode = tempReply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toUInt();
  if (httpstatuscode == 302) {
    currentReply =
        networkManager->get(QNetworkRequest(tempReply->attribute(QNetworkRequest::RedirectionTargetAttribute).toUrl()));
    connect(currentReply, &QNetworkReply::downloadProgress, this, &Downloader::downloadProgress);
    connect(currentReply, &QNetworkReply::readyRead, this, &Downloader::downloadReadyRead);
  } else if (httpstatuscode == 200) {
    outputFile.reset();  // Delete the temporary file object to make the file openable
    currentReply = nullptr;
  } else {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        QString("Error %0 while downloading the file: %1")
            .arg(httpstatuscode)
            .arg(QString(tempReply->attribute(QNetworkRequest::HttpReasonPhraseAttribute).toByteArray())));
  }
  tempReply->deleteLater();
}

}  // namespace Helper
}  // namespace VideoStitch
