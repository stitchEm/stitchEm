// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef GENERICTABLEWIDGET_HPP
#define GENERICTABLEWIDGET_HPP

#include <QTableWidget>

class GenericTableWidget : public QTableWidget {
  Q_OBJECT
 public:
  explicit GenericTableWidget(QWidget* const parent = nullptr);

  void addElementToTable(QWidget* widget);
  void clearElements();
  void setResizable();

  void initializeTable(int nbColumns, int nbRows);
  void finishTable();

 private:
  int currentCol;
  int currentRow;
};

#endif  // GENERICTABLEWIDGET_H
