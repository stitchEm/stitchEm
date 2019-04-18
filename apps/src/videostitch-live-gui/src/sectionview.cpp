// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "sectionview.hpp"

#include "sectionmodel.hpp"

#include <QHeaderView>
#include <QPushButton>

SectionView::SectionView(QWidget* parent) : QTableView(parent) {
  setSelectionBehavior(QAbstractItemView::SelectRows);
  setShowGrid(false);
  setProperty("vs-section-container", true);
  horizontalHeader()->setVisible(false);
  verticalHeader()->setVisible(false);
  connect(&mapperForDelete, static_cast<void (QSignalMapper::*)(const QString&)>(&QSignalMapper::mapped), this,
          &SectionView::askToDeleteRow);
}

void SectionView::setSectionModel(SectionModel* model) {
  setModel(model);
  horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);

  updateDeleteButtons();
  // This signal should be set AFTER 'setModel' (which also update the GUI)
  connect(model, &QAbstractItemModel::modelReset, this, &SectionView::updateDeleteButtons);
}

void SectionView::updateDeleteButtons() {
  const SectionModel* myModel = qobject_cast<const SectionModel*>(model());
  int idColumnIndex = myModel->getIdColumnIndex();
  int deleteColumnIndex = myModel->getDeleteColumnIndex();
  if (!myModel || idColumnIndex == -1 || deleteColumnIndex == -1) {
    return;
  }

  int nbRows = myModel->rowCount();
  for (int row = 0; row < nbRows; ++row) {
    QString id = myModel->data(myModel->index(row, idColumnIndex)).toString();

    QModelIndex modelIndexForDelete = myModel->index(row, deleteColumnIndex);
    QPushButton* button = new QPushButton(this);
    button->setProperty("vs-section-delete-button", true);
    button->setToolTip(myModel->getDeleteButtonTooltip());
    connect(button, &QPushButton::clicked, &mapperForDelete,
            static_cast<void (QSignalMapper::*)()>(&QSignalMapper::map));
    mapperForDelete.setMapping(button, id);
    setIndexWidget(modelIndexForDelete, button);
  }
}
