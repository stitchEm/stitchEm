// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QLocalServer>

class VSLocalSocket;

// unique identifier to ensure there is only one instance running
static const char VSSTUDIOKEY[] =
    "C46F88B1-1FBD-4280-AA22-F358AFB019BD";  // if you change this key, also modify the Windows installer script mutex
static const char VSLIVEKEY[] = "52CB03D5-4269-487C-A4E6-CA059D4C1BF0";
static const char VSBATCHKEY[] = "F3163C6A-76F4-416E-8518-6CD13CB1C798";

/**
 * @brief Local server used to send/Receive messages accross the multiple instances of VideoStitch.
 */
class VSLocalServer : public QLocalServer {
  Q_OBJECT

 public:
  explicit VSLocalServer(QObject* parent = nullptr);

  /**
   * @brief Tries to start the local server with a give name (acts like a ports for TCP servers)
   * @param name Name on which you want the server to listend
   * @return true = the server started successfully, false it did not.
   */
  bool tryToStartServer(QString name);

 signals:
  /**
   * @brief Signal emitted when a noew clients has just connected
   */
  void newClient(VSLocalSocket*);

 private:
  /**
   * @brief Starts the local server with a give name (acts like a ports for TCP servers)
   * @param name Name on which you want the server to listend
   * @return true = the server started successfully, false it did not.
   */
  bool startServer(QString name);

 private slots:
  /**
   * @brief Slot called when a new connections is being processed.
   */
  void on_newConnection();
};
