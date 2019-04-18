// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QAbstractTableModel>

class SectionModel : public QAbstractTableModel {
  Q_OBJECT

 public:
  explicit SectionModel(QObject* parent = nullptr);
  virtual ~SectionModel() = default;

  int getIdColumnIndex() const;
  int getDeleteColumnIndex() const;
  QString getDeleteButtonTooltip() const;

  // reimplemented methods
  virtual QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const;

 protected:
  // Set these data if you want delete buttons
  int idColumnIndex = -1;
  int deleteColumnIndex = -1;
  QString deleteButtonTooltip;

 private:
  Q_DISABLE_COPY(SectionModel)
};
