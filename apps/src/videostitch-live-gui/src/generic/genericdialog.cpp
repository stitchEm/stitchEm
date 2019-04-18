// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "genericdialog.hpp"
#include "guiconstants.hpp"
#include "dialogbackground.hpp"
#include "animations/fadeanimation.hpp"

#include "libvideostitch-gui/mainwindow/LibLogHelpers.hpp"

#include <QKeyEvent>

void GenericDialog::createAcceptDialog(const QString& title, const QString& message, QWidget* const parent) {
  GenericDialog* dialog = new GenericDialog(title, message, GenericDialog::DialogMode::ACCEPT, parent);
  dialog->show();
}

GenericDialog::GenericDialog(const QString& title, const QString& message, DialogMode mode, QWidget* const parent)
    : QFrame(parent),
      background(new DialogBackground(parent)),
      animationDialog(new AnimatedWidget()),
      animationBackground(new AnimatedWidget()) {
  parent->installEventFilter(this);
  setupUi(this);
  buttonAccept->setProperty("vs-button-medium", true);
  buttonCancel->setProperty("vs-button-medium", true);
  buttonNext->setProperty("vs-button-medium", true);
  background->setObjectName("transparentBackground");
  background->hide();

  setWindowFlags(windowFlags() & ~Qt::WindowMinMaxButtonsHint & ~Qt::WindowContextHelpButtonHint);
  setAttribute(Qt::WA_DeleteOnClose);

  connect(buttonCancel, &QPushButton::clicked, this, &GenericDialog::notifyCancelClicked);
  connect(buttonAccept, &QPushButton::clicked, this, &GenericDialog::notifyAcceptClicked);
  connect(buttonAccept, &QPushButton::clicked, this, &GenericDialog::close);

  setDialogMode(mode);
  labelDialogTitle->setText(title);
  labelDialogMessage->setText(message);
  animationBackground->installAnimation(new FadeAnimation(this));
  animationDialog->installAnimation(new FadeAnimation(background));
  if (parent != nullptr) updatePosition(parent->width(), parent->height());
}

GenericDialog::GenericDialog(const VideoStitch::Status& status, QWidget* const parent)
    : GenericDialog(VideoStitch::Helper::createTitle(status), VideoStitch::Helper::createErrorBacktrace(status),
                    DialogMode::ACCEPT, parent) {}

GenericDialog::GenericDialog(const QString& title, const VideoStitch::Status& status, QWidget* const parent)
    : GenericDialog(title, VideoStitch::Helper::createErrorBacktrace(status), DialogMode::ACCEPT, parent) {}

GenericDialog::~GenericDialog() {}

void GenericDialog::updatePosition(unsigned int parentWidth, unsigned int parentHeight) {
  move((parentWidth - width()) / 2, (parentHeight - height()) / 2);
  background->setFixedSize(parentWidth, parentHeight);
}

void GenericDialog::setDialogMode(const DialogMode mode) {
  switch (mode) {
    case DialogMode::ACCEPT:
      buttonAccept->setVisible(true);
      buttonCancel->setVisible(false);
      buttonNext->setVisible(false);
      break;
    case DialogMode::ACCEPT_CANCEL:
      buttonAccept->setVisible(true);
      buttonCancel->setVisible(true);
      buttonNext->setVisible(false);
      break;
    case DialogMode::NEXT:
      buttonAccept->setVisible(false);
      buttonCancel->setVisible(false);
      buttonNext->setVisible(true);
      break;
  }
}

QString GenericDialog::getTitle() const { return labelDialogTitle->text(); }

QString GenericDialog::getMessage() const { return labelDialogMessage->text(); }

void GenericDialog::showEvent(QShowEvent*) {
  background->show();
  background->raise();
  show();
  raise();
  animationBackground->startArrivalAnimations();
  animationDialog->startArrivalAnimations();
}

void GenericDialog::closeEvent(QCloseEvent*) {
  animationBackground->startDepartureAnimations();
  animationBackground->startDepartureAnimations();
  background->close();
  close();
}

void GenericDialog::hideEvent(QHideEvent*) {
  animationBackground->startDepartureAnimations();
  animationBackground->startDepartureAnimations();
}

void GenericDialog::resizeEvent(QResizeEvent*) { updatePosition(parentWidget()->width(), parentWidget()->height()); }

void GenericDialog::keyReleaseEvent(QKeyEvent* event) {
  if (event->key() == Qt::Key_Return) {
    buttonAccept->click();
  } else if (event->key() == Qt::Key_Escape) {
    buttonCancel->click();
  } else {
    QFrame::keyReleaseEvent(event);
  }
}

bool GenericDialog::eventFilter(QObject* object, QEvent* event) {
  if (event->type() == QEvent::Resize) {
    updatePosition(parentWidget()->width(), parentWidget()->height());
  }
  return QWidget::eventFilter(object, event);
}
