// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "packet.hpp"

class QLocalSocket;

/**
 * @brief Class which encapsulate a QlocalSocket class and deals with our Packet class
 */
class VSLocalSocket : public QObject {
  Q_OBJECT

 public:
  explicit VSLocalSocket(QObject *parent = nullptr);
  /**
   * @brief Sets the QLocalSocket to wrap
   * @param pointer to the QLocalSocket
   */
  void setSocket(QLocalSocket *socket);

 signals:
  /**
   * @brief Signal emitted when a pcket has been received
   */
  void messageReceived(Packet);
  /**
   * @brief Signal emitted when the socket has just connected to the server
   */
  void connected();
  /**
   * @brief Signal emitted when the socket has just disconnected to the server
   */
  void disconnected();

 private slots:
  /**
   * @brief Slot called when the socket has just connected to the server
   */
  void socketConnected();
  /**
   * @brief Slot called when the socket has just disconnected from the server
   */
  void socketDisconnected();
  /**
   * @brief Slot called when a message is ready to be read
   */
  void messageAvailable();

 private:
  QLocalSocket *socket;
};
