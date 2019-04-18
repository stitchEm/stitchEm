// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "vslocalserver.hpp"

#include "vslocalsocket.hpp"

#include <QLocalSocket>

VSLocalServer::VSLocalServer(QObject *parent) : QLocalServer(parent) {}

bool VSLocalServer::tryToStartServer(QString name) {
  bool ret = startServer(name);
  if (!ret) {
    QLocalSocket ping;
    ping.connectToServer(name);
    if (ping.waitForConnected(100)) {
      // the other application is listening, this one is not unique
      return false;
    } else {
      if (ping.error() == QLocalSocket::ConnectionRefusedError) {
        // other server is dead, possibly from a crashed application. kill it.
        QLocalServer::removeServer(name);
        // try again
        ret = startServer(name);
      }
    }
  }
  return ret;
}

bool VSLocalServer::startServer(QString name) {
  connect(this, &VSLocalServer::newConnection, this, &VSLocalServer::on_newConnection, Qt::UniqueConnection);
  return listen(name);
}

void VSLocalServer::on_newConnection() {
  VSLocalSocket *newConnection = new VSLocalSocket(this);
  newConnection->setSocket(nextPendingConnection());
  emit newClient(newConnection);
}
