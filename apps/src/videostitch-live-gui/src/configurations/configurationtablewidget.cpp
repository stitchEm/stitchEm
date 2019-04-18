// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "configurationtablewidget.hpp"

static const QString& ICON_PREFIX_NAME("icon");

ConfigurationTableWidget::ConfigurationTableWidget(ConfigIdentifier identifier, const QString& title,
                                                   QWidget* const parent)
    : QWidget(parent), configIdentifier(identifier) {
  setupUi(this);
  labelItemTitle->setText(title);
  // TODO FIXME: title is a translatable text. Our icon name use the english version of title
  // So this will not work in another language
  iconConfig->setObjectName(ICON_PREFIX_NAME + title);
}

ConfigurationTableWidget::~ConfigurationTableWidget() {}

ConfigIdentifier ConfigurationTableWidget::getConfigIdentifier() const { return configIdentifier; }
