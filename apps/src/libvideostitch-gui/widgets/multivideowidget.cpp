// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "multivideowidget.hpp"

#include "libvideostitch-gui/utils/sourcewidgetlayoututil.hpp"

#include <QThread>

static const QColor clearColor(0x12, 0x12, 0x12);

MultiVideoWidget::MultiVideoWidget(QWidget *parent) : GenericVideoWidget(parent) {}

MultiVideoWidget::~MultiVideoWidget() {}

void MultiVideoWidget::syncOn() { sync = true; }

void MultiVideoWidget::syncOff() { sync = false; }

void MultiVideoWidget::render(std::shared_ptr<VideoStitch::Core::SourceOpenGLSurface> surf, mtime_t date) {
  {
    std::lock_guard<std::mutex> lock(textureMutex);
    textures[(int)surf->sourceId] = surf;
  }
  if (surf->sourceId == 0) {
    if (sync) {
      mtime_t sleep = date - ref_vs - (mtime_t)(clk.nsecsElapsed() / 1000);
      if (sleep < 0 || sleep > 200000) {
        // absurd durations indicates that we're not playing
        // reset the clock
        ref_vs = date;
        clk.restart();
      } else if (sleep > 10000) {       // don't bother sleeping for less than 10 ms
        QThread::usleep(sleep - 2000);  // keep a 2 ms margin just in case
      }
    }

    emit gotFrame(date);
  }
}

void MultiVideoWidget::paintGL() {
  glClearColor(clearColor.redF(), clearColor.greenF(), clearColor.blueF(), clearColor.alphaF());
  glClear(GL_COLOR_BUFFER_BIT);

  const qreal retinaScale = devicePixelRatio();
  std::lock_guard<std::mutex> lock(textureMutex);

  const int nbInputs = int(textures.size());
  int maxTexWidth = 0;
  int maxTexHeight = 0;
  for (int index = 0; index < nbInputs; ++index) {
    if (textures[index] != nullptr) {
      maxTexWidth = qMax(maxTexWidth, int(textures[index]->getWidth()));
      maxTexHeight = qMax(maxTexHeight, int(textures[index]->getHeight()));
    }
  }
  if (maxTexWidth == 0 || maxTexHeight == 0) {
    return;
  }

  const int nbColumns = SourceWidgetLayoutUtil::getColumnsNumber(nbInputs);
  const int nbLines = SourceWidgetLayoutUtil::getLinesNumber(nbInputs);
  const int additionalMargin = 10 * retinaScale;
  const int viewWidth = width() * retinaScale;
  const int viewHeight = height() * retinaScale;
  const int availableViewWidth = viewWidth - additionalMargin * nbColumns;
  const int availableViewHeight = viewHeight - additionalMargin * nbLines;
  const float verticalMargin =
      (qMax(0.f, availableViewWidth - maxTexWidth * nbColumns * availableViewHeight / float(maxTexHeight * nbLines)) +
       additionalMargin * nbColumns) /
      2.0;
  const float horizontalMargin =
      (qMax(0.f, availableViewHeight - maxTexHeight * nbLines * availableViewWidth / float(maxTexWidth * nbColumns)) +
       additionalMargin * nbLines) /
      2.0;

  for (int index = 0; index < nbInputs; ++index) {
    if (textures[index] != nullptr) {
      const int lineIndex = SourceWidgetLayoutUtil::getItemLine(index, nbColumns);
      const int columnIndex = SourceWidgetLayoutUtil::getItemColumn(index, nbColumns);
      const int reversedLineIndex = nbLines - lineIndex - 1;
      const int x = columnIndex * viewWidth / nbColumns;
      const int y = reversedLineIndex * viewHeight / nbLines;
      glViewport(x, y, viewWidth / nbColumns, viewHeight / nbLines);
      paintFrame(textures[index]->texture, viewWidth, viewHeight, verticalMargin, horizontalMargin);
    }
  }
}
