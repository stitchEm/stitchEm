// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef YOUTUBETABLEVIEW_HPP
#define YOUTUBETABLEVIEW_HPP

#include <QTableView>
#include <QScopedPointer>
#include <QAbstractItemModel>

class YoutubeTableView : public QTableView {
  Q_OBJECT

 public:
  explicit YoutubeTableView(QWidget* parent = 0);
  ~YoutubeTableView();

  void setModel(QAbstractItemModel* newModel);

 private:
  void updateTableSize();
};

#endif  // YOUTUBETABLEVIEW_HPP
