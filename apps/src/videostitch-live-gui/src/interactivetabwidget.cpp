// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "interactivetabwidget.hpp"
#include "outputcontrolspanel.hpp"
#include "guiconstants.hpp"
#include "widgetsmanager.hpp"

#include "libvideostitch-gui/widgets/deviceinteractivewidget.hpp"

#include <QShortcut>
#include <QWindow>
#include <QDesktopWidget>

InteractiveTabWidget::InteractiveTabWidget(QWidget* const parent) : QFrame(parent) {
  setupUi(this);
  interactiveWidget->syncOff();
  fullScreenShortCut = new QShortcut(QKeySequence(Qt::Key_F), interactiveWidget);
  connect(fullScreenShortCut, &QShortcut::activated, this, &InteractiveTabWidget::onFullScreenActivated);
}

InteractiveTabWidget::~InteractiveTabWidget() {}

void InteractiveTabWidget::setOutputWidgetReference(OutputControlsPanel* buttonsBar) {
  outputsPanel = buttonsBar;
  outputsPanel->configure(OutputControlsPanel::HideOrientation | OutputControlsPanel::HidePanorama |
                          OutputControlsPanel::HideSnapshot | OutputControlsPanel::HideAudioProcessors);
  horizontalLayout->insertWidget(1, outputsPanel);
}

OutputControlsPanel* InteractiveTabWidget::getControlsBar() const { return outputsPanel; }

DeviceInteractiveWidget* InteractiveTabWidget::getInteractiveWidget() const { return interactiveWidget; }

void InteractiveTabWidget::onFullScreenActivated() {
  if (interactiveWidget->isFullScreen()) {
    interactiveWidget->setParent(this);
    layoutOutputControl->insertWidget(0, interactiveWidget);
    interactiveWidget->show();
    interactiveWidget->update();
    // Recover focus after fullscreen off
    QApplication::setActiveWindow(WidgetsManager::getInstance()->getMainWindowRef());
  } else {
    layoutOutputControl->removeWidget(interactiveWidget);
    interactiveWidget->setParent(nullptr);
    interactiveWidget->show();
    // Show the fullscreen widget in the same screen as the containing widget
    QWidget* screen = QApplication::desktop()->screen(QApplication::desktop()->screenNumber(this));
    interactiveWidget->windowHandle()->setScreen(screen->windowHandle()->screen());
    interactiveWidget->showFullScreen();
  }
}
