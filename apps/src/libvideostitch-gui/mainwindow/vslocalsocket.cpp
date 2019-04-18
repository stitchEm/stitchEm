// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "vslocalsocket.hpp"

#include <QLocalSocket>

VSLocalSocket::VSLocalSocket(QObject *parent) : QObject(parent), socket(nullptr) {}

void VSLocalSocket::setSocket(QLocalSocket *sock) {
  socket = sock;
  connect(socket, &QLocalSocket::connected, this, &VSLocalSocket::socketConnected);
  connect(socket, &QLocalSocket::disconnected, this, &VSLocalSocket::socketDisconnected);
  connect(socket, &QLocalSocket::readyRead, this, &VSLocalSocket::messageAvailable);
  socket->setParent(this);
}

void VSLocalSocket::messageAvailable() {
  QByteArray packet;
  QDataStream in(&packet, QIODevice::ReadOnly);
  packet = socket->readAll();
  Packet message;
  in >> message;
  emit messageReceived(message);
}

void VSLocalSocket::socketConnected() { emit connected(); }

void VSLocalSocket::socketDisconnected() {
  emit disconnected();
  deleteLater();
}
