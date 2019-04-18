// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef GENERICLOADER_HPP
#define GENERICLOADER_HPP

#include <QFrame>

class QLabel;
class QMovie;

/**
 * @brief A frame showing a loading animation that must be shown during the execution of lib process
 */
class GenericLoader : public QFrame {
  Q_OBJECT
 public:
  /**
   * @brief Constructor
   * @param message An message to be shown
   * @param parent Parent widget
   */
  explicit GenericLoader(const QString& message, QWidget* const parent = nullptr);

  ~GenericLoader();

  /**
   * @brief Updates the frame size
   * @param width
   * @param height
   */
  void updateSize(unsigned int width, unsigned int height);

  /**
   * @brief Changes the central image
   * @param image
   */
  void changeCentralImage(const QString& image);

 protected:
  virtual bool eventFilter(QObject*, QEvent*);

 private:
  QScopedPointer<QMovie> movieLoading;

  QLabel* labelLoading;

  QLabel* labelMessage;

  QLabel* labelPercentage;

 protected:
  void showEvent(QShowEvent*);
};

#endif  // GENERICLOADER_HPP
