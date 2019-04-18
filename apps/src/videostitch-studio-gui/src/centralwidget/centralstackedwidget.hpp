// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "outputtabwidget.hpp"
#include "processtabwidget.hpp"
#include "widgets/interactivetabwidget.hpp"
#include "libvideostitch-gui/centralwidget/sourcewidget.hpp"
#include "libvideostitch-gui/widgets/welcome/welcomescreenwidget.hpp"
#include <QWidget>

/**
 * @brief Central stack widget used to display the different modes.
 */
class CentralStackedWidget : public QWidget {
  Q_OBJECT
 public:
  /**
   * @brief Lists the widgets of the stacked widget.
   */
  enum VSTabWidget { source = 0, output = 1, interactive = 2, process = 3, welcome = 4, undefined = 0xFF };

  explicit CentralStackedWidget(QWidget* parent = nullptr);

  VSTabWidget activeTab() const;
  bool allowsPlayback();
  void setPreviewFullScreen(bool activate);
  SourceWidget* getSourceTabWidget();
  OutputTabWidget* getOutputTabWidget();
  InteractiveTabWidget* getInteractiveTabWidget();
  ProcessTabWidget* getProcessTabWidget();
  WelcomeScreenWidget* getWelcomeTabWidget();

 public slots:
  void activate(VSTabWidget index);

 private:
  QWidget* getTabWidget(const CentralStackedWidget::VSTabWidget index);
  void deactivate(VSTabWidget index);
  VSTabWidget _currentIndex;
  SourceWidget sourceWidget;
  OutputTabWidget outputWidget;
  InteractiveTabWidget interactiveWidget;
  WelcomeScreenWidget welcomeWidget;
  ProcessTabWidget processTabWidget;
};
