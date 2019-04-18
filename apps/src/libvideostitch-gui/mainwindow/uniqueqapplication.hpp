// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "packet.hpp"

#include <QApplication>

class VSLocalSocket;

/**
 * @brief Class used to start a unique application.
 *        Starting the application with this class will ensure that there is only one running instance.
 *        If another instance is running, the application will exit.
 */
class VS_GUI_EXPORT UniqueQApplication : public QApplication {
  Q_OBJECT

 public:
  UniqueQApplication(int &argc, char *argv[], QString applicationKey = "");
  ~UniqueQApplication();

  static UniqueQApplication *instance();
  static void initializeOrganization();

  /**
   * @brief uniqueInstance Method that returns whether the application is unique or not
   * @return true = it is unique / false = it is not
   */
  bool uniqueInstance() const;

  void setUpLogger();

  /**
   * @brief cleanup Function used to cleanup the application when it closes/crashes
   */
  void cleanup();

  void loadStylesheetFile(QString stylesheetPath, QString stylesheetVariablesPath = QString(),
                          QString commonStylesheetVariablesPath = QString());
  void loadTranslationFiles();

  QString getYoutubeUrl() const;
  QString getTutorialUrl() const;
  void setYoutubeUrl(QString url);
  void setTutorialUrl(QString url);

 public slots:
  /**
   * @brief connectAndSend Connects the app to the local server and sends a packet to it.
   * @param pack Packet to send
   */
  void connectAndSend(const Packet &pack, QString host, bool andDie = true);
  /**
   * @brief receiveMessage Slot called when a message has been received
   * @param packet Received message to process
   */
  void receiveMessage(const Packet &packet);
  /**
   * @brief incomingClient Slot called when a new version of VideoStitch has just connected to the local server
   * @param instance Socket of the new client
   */
  void incomingClient(VSLocalSocket *instance);
  /**
   * @brief sendPacket Sends a packet to another instance of VideoStitch
   */
  void sendPacket();

  void closeSocket();
  void restartApplication();

 signals:
  void messageAvailable(Packet packet);
  /**
   * @brief reqOpenFile Sends a signal to the mainwindow to ask it to open a file
   * @param filepath file to open
   */
  void reqOpenFile(const QStringList &filepath);

 private:
  /**
   * @brief event Inherited from QObject::Event, this function is called when a QObject receives an event
   * @param event Event to process
   * @return true if the event has been accepted and process, false instead
   */
  bool event(QEvent *event);
  void setUpLocalServer();

  class Impl;
  QScopedPointer<Impl> impl;
};
