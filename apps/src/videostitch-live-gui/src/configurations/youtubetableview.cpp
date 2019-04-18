// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QHeaderView>

#include "youtubetableview.hpp"
#include "guiconstants.hpp"

YoutubeTableView::YoutubeTableView(QWidget* parent) : QTableView(parent) {
  horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
  verticalHeader()->setDefaultSectionSize(YOUTUBE_ICON_HEIGHT + 10);
}

void YoutubeTableView::updateTableSize() {
  auto height = model()->rowCount() * rowHeight(0) + 2;
  resize(width(), height);

  if (model()->columnCount()) {
    horizontalHeader()->setSectionResizeMode(0, QHeaderView::Fixed);
    horizontalHeader()->resizeSection(0, 2 * YOUTUBE_ICON_HEIGHT);

    horizontalHeader()->setSectionResizeMode(3, QHeaderView::Fixed);
    horizontalHeader()->resizeSection(3, 3 * YOUTUBE_PRIVACY_WIDTH);
  }
}

YoutubeTableView::~YoutubeTableView() {}

void YoutubeTableView::setModel(QAbstractItemModel* newModel) {
  QTableView::setModel(newModel);
  connect(newModel, &QAbstractItemModel::modelReset, this, &YoutubeTableView::updateTableSize);
}
