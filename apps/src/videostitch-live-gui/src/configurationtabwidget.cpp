// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "guiconstants.hpp"
#include "configurationtabwidget.hpp"
#include "configurations/configoutputswidget.hpp"
#include "configurations/configpanorama.hpp"
#include "configurations/configexposurewidget.hpp"
#include "videostitcher/liveprojectdefinition.hpp"
#include "videostitcher/livestitchercontroller.hpp"

#include "libvideostitch-gui/mainwindow/outputfilehandler.hpp"

ConfigurationTabWidget::ConfigurationTabWidget(QWidget* const parent)
    : QFrame(parent), tableWidget(new GenericTableWidget(this)), projectDefinition(nullptr) {
  setupUi(this);
  labelConfigurationTitle->setProperty("vs-title1", true);
  initializePageMap();
  tableWidget->setObjectName("configSectionTable");
  mainPageLayout->addWidget(tableWidget);
  addConfigValues();
  connect(tableWidget, &GenericTableWidget::cellClicked, this, &ConfigurationTabWidget::onConfigItemSelected);
}

ConfigurationTabWidget::~ConfigurationTabWidget() {}

void ConfigurationTabWidget::restore() {
  changeConfigurationPage(ConfigIdentifier::CONFIG_MAIN);
  getConfigOutputs()->restoreMainPage();
}

void ConfigurationTabWidget::initializePageMap() {
  pageMap[ConfigIdentifier::CONFIG_MAIN] = pageMainConfiguration;  // not an item
}

void ConfigurationTabWidget::addConfigValues() {
  tableWidget->initializeTable(CONFPANEL_MAX_COLS, CONFPANEL_MAX_ROWS);
  addSingleConfigurationValue(ConfigIdentifier::CONFIG_OUTPUT, tr("OUTPUTS"), new ConfigOutputsWidget(this));
  tableWidget->finishTable();
}

void ConfigurationTabWidget::addSingleConfigurationValue(ConfigIdentifier identifier, const QString& title,
                                                         IConfigurationCategory* widget, bool addInTable) {
  if (addInTable) {
    tableWidget->addElementToTable(new ConfigurationTableWidget(identifier, title, pageMainConfiguration));
  }
  connect(widget, &IConfigurationCategory::saved, this, &ConfigurationTabWidget::onFileHasTobeSaved);
  connect(widget, &IConfigurationCategory::reqBack, this, &ConfigurationTabWidget::showMainPage);
  connect(this, &ConfigurationTabWidget::injectProject, widget, &IConfigurationCategory::setProject);
  connect(this, &ConfigurationTabWidget::projectCleared, widget, &IConfigurationCategory::clearProject);

  pageMap[identifier] = widget;
  stackedWidget->addWidget(widget);
}

void ConfigurationTabWidget::changeConfigurationPage(const ConfigIdentifier identifier) {
  stackedWidget->setCurrentWidget(pageMap.value(identifier));
}

void ConfigurationTabWidget::onConfigItemSelected(int row, int col) {
  ConfigurationTableWidget* widget(qobject_cast<ConfigurationTableWidget*>(tableWidget->cellWidget(row, col)));
  if (widget != nullptr) {
    changeConfigurationPage(widget->getConfigIdentifier());
  }
}

void ConfigurationTabWidget::onFileHasTobeSaved() {
  emit reqSaveProject(ProjectFileHandler::getInstance()->getFilename());
  emit reqStitcherReload();
}

void ConfigurationTabWidget::showMainPage() { changeConfigurationPage(ConfigIdentifier::CONFIG_MAIN); }

void ConfigurationTabWidget::setProject(ProjectDefinition* project) {
  projectDefinition = qobject_cast<LiveProjectDefinition*>(project);
  emit injectProject(projectDefinition);
}

void ConfigurationTabWidget::clearProject() {
  projectDefinition = nullptr;
  emit projectCleared();
}

void ConfigurationTabWidget::toggleOutput(const QString& id) { getConfigOutputs()->toggleOutput(id); }

ConfigOutputsWidget* ConfigurationTabWidget::getConfigOutputs() const {
  return qobject_cast<ConfigOutputsWidget*>(pageMap.value(ConfigIdentifier::CONFIG_OUTPUT));
}
