// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QSignalMapper>
#include <QTableView>

class SectionModel;

class SectionView : public QTableView {
  Q_OBJECT

 public:
  explicit SectionView(QWidget* parent = nullptr);
  virtual ~SectionView() = default;

  void setSectionModel(SectionModel* model);

 signals:
  void askToDeleteRow(QString rowId);

 private:
  Q_DISABLE_COPY(SectionView)
  void updateDeleteButtons();

  QSignalMapper mapperForDelete;
};
