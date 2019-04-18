// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef GENERICDIALOG_HPP
#define GENERICDIALOG_HPP

#include <QFrame>
#include <QStackedWidget>
#include "ui_genericdialog.h"
#include "animations/animatedwidget.hpp"

#include <libvideostitch/status.hpp>

class DialogBackground;
/**
 * @brief A generic dialog for showing alerts, user decisions, and small configuration steps
 */
class GenericDialog : public QFrame, public Ui::GenericDialogClass {
  Q_OBJECT
 public:
  enum class DialogMode {
    ACCEPT,         // Shows only accept button
    ACCEPT_CANCEL,  // Shows accept and cancel buttons
    NEXT            // Shows next button (for steps)
  };

  /**
   * @brief Creates a generic dialog (with accept button only) and shows it
   * @param title A top widget title
   * @param message A message
   * @param parent Parent widget
   */
  static void createAcceptDialog(const QString& title, const QString& message, QWidget* const parent);

  /**
   * @brief Constructor
   * @param title A top widget title
   * @param message A message
   * @param mode A dialog mode defined in DialogMode enum
   * @param parent Parent widget
   */
  explicit GenericDialog(const QString& title, const QString& message, DialogMode mode, QWidget* const parent);

  /**
   * @brief Constructor
   * @param status An error Status
   */
  explicit GenericDialog(const VideoStitch::Status& status, QWidget* const parent);

  /**
   * @brief Constructor
   * @param title A top widget title
   * @param status An error Status
   */
  explicit GenericDialog(const QString& title, const VideoStitch::Status& status, QWidget* const parent);

  /**
   * @brief Destructor
   */
  ~GenericDialog();

  /**
   * @brief Changes the current size (used for main window resizing event)
   * @param width Desired width
   * @param height Desired height
   */
  void updatePosition(unsigned int width, unsigned int height);

  /**
   * @brief Change the dialog mode once the dialog is constructed
   * @param mode A Dialog mode
   */
  void setDialogMode(const DialogMode mode);

  QString getTitle() const;
  QString getMessage() const;

 signals:
  void notifyAcceptClicked();
  void notifyCancelClicked();

 protected:
  DialogBackground* background;

  virtual void showEvent(QShowEvent*);
  virtual void closeEvent(QCloseEvent*);
  virtual void hideEvent(QHideEvent*);
  virtual void resizeEvent(QResizeEvent*);
  virtual void keyReleaseEvent(QKeyEvent* event);

  virtual bool eventFilter(QObject*, QEvent*);

 private:
  QScopedPointer<AnimatedWidget> animationDialog;
  QScopedPointer<AnimatedWidget> animationBackground;
};

#endif  // GENERICDIALOG_HPP
