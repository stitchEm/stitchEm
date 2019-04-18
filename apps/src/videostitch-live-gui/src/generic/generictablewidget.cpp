// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QHeaderView>
#include "generictablewidget.hpp"

GenericTableWidget::GenericTableWidget(QWidget* const parent) : QTableWidget(parent), currentCol(0), currentRow(0) {
  viewport()->setFocusPolicy(Qt::NoFocus);
  setGridStyle(Qt::NoPen);
  setFrameShape(QFrame::NoFrame);
  setEditTriggers(QAbstractItemView::AnyKeyPressed);
  setTabKeyNavigation(false);
  setProperty("showDropIndicator", QVariant(false));
  setDragDropOverwriteMode(false);
  setSelectionMode(QAbstractItemView::NoSelection);
  setShowGrid(false);
  setCornerButtonEnabled(false);
  horizontalHeader()->setVisible(false);
  horizontalHeader()->setHighlightSections(false);
  verticalHeader()->setVisible(false);
}

void GenericTableWidget::addElementToTable(QWidget* widget) {
  widget->setFocusPolicy(Qt::NoFocus);
  setCellWidget(currentRow, currentCol, widget);

  if (currentCol < columnCount() - 1) {
    ++currentCol;
  } else {
    currentCol = 0;
    ++currentRow;
  }
}

void GenericTableWidget::clearElements() {
  clear();
  currentCol = 0;
  currentRow = 0;
  setColumnCount(0);
  setRowCount(0);
}

void GenericTableWidget::setResizable() { horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch); }

void GenericTableWidget::initializeTable(int nbColumns, int nbRows) {
  setColumnCount(nbColumns);
  setRowCount(nbRows);
}

void GenericTableWidget::finishTable() {
  resizeColumnsToContents();
  resizeRowsToContents();
}
