// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef VSLISTWIDGET_HPP
#define VSLISTWIDGET_HPP

#include <QListWidget>
/**
 * @brief The VSListWidget class is a QListWidget which automatically constrains to fit its content
 *  when using the function addItemAndFitToContent.
 */
class VS_GUI_EXPORT VSListWidget : public QListWidget {
  Q_OBJECT
 public:
  explicit VSListWidget(QWidget *parent = 0) : QListWidget(parent) {
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  }

  virtual QSize sizeHint() const {
    int maxWidth = 0;
    for (int i = 0; i < count(); i++) {
      if (sizeHintForColumn(0) > maxWidth) maxWidth = sizeHintForColumn(0);
    }
    // all the lines should have the same height
    return QSize(maxWidth + 100, (sizeHintForRow(0) + 2) * count());
  }
  /**
   * @brief Adds an item to the list and resizes to widget to fit its content.
   * @param item Item to add.
   */
  void addItem(QListWidgetItem *item) {
    QListWidget::addItem(item);
    setMaximumSize(sizeHint());
    setMinimumSize(sizeHint());
    resize(sizeHint());
  }
};

#endif  // VSLISTWIDGET_HPP
