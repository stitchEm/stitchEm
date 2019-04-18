// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "sectionmodel.hpp"

#include "guiconstants.hpp"

SectionModel::SectionModel(QObject* parent) : QAbstractTableModel(parent) {}

int SectionModel::getIdColumnIndex() const { return idColumnIndex; }

int SectionModel::getDeleteColumnIndex() const { return deleteColumnIndex; }

QString SectionModel::getDeleteButtonTooltip() const { return deleteButtonTooltip; }

QVariant SectionModel::data(const QModelIndex& index, int role) const {
  if (role == Qt::SizeHintRole) {
    int deleteColumnIndex = getDeleteColumnIndex();
    if (deleteColumnIndex != -1 && index.column() == deleteColumnIndex) {
      return QSize(ITEM_HEIGHT - 2 * ITEM_VERTICAL_BORDER, ITEM_HEIGHT);
    } else {
      return QSize(0, ITEM_HEIGHT);
    }
  }
  return QVariant();
}
