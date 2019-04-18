// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef GPULISTWIDGET_HPP
#define GPULISTWIDGET_HPP

#include "../widgets/vslistwidget.hpp"

class VS_GUI_EXPORT GpuListWidget : public VSListWidget {
  Q_OBJECT

 public:
  explicit GpuListWidget(QWidget *parent = nullptr);
  ~GpuListWidget();

  QVector<int> getSelectedGpus() const;

  void setSelectedGpus(QVector<int> selectedGpus);

 signals:
  void selectedGpusChanged();

 private slots:
  void toggleItemCheckState(QListWidgetItem *item);
  void checkGpus();
  void checkGpusFor(QListWidgetItem *changedItem);
};

#endif  // GPULISTWIDGET_HPP
