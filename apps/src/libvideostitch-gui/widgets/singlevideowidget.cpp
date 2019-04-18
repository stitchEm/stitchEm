// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QThread>
#include "singlevideowidget.hpp"

static const QColor clearColor(0x2c, 0x2c, 0x2c);

SingleVideoWidget::SingleVideoWidget(const int id, QWidget *parent) : GenericVideoWidget(parent), inputId(id) {}

SingleVideoWidget::~SingleVideoWidget() {}

void SingleVideoWidget::render(std::shared_ptr<VideoStitch::Core::SourceOpenGLSurface> surf, mtime_t date) {
  {
    std::lock_guard<std::mutex> lock(textureMutex);
    textures[(int)surf->sourceId] = surf;
  }

  mtime_t sleep = date - ref_vs - (mtime_t)(clk.nsecsElapsed() / 1000);
  if (sleep < 0 || sleep > 200000) {
    // absurd durations indicates that we're not playing
    // reset the clock
    ref_vs = date;
    clk.restart();
  } else if (sleep > 10000) {       // don't bother sleeping for less than 10 ms
    QThread::usleep(sleep - 2000);  // keep a 2 ms margin just in case
  }
  emit gotFrame(date);
}

void SingleVideoWidget::paintGL() {
  glClearColor(clearColor.redF(), clearColor.greenF(), clearColor.blueF(), clearColor.alphaF());
  glClear(GL_COLOR_BUFFER_BIT);

  std::lock_guard<std::mutex> lock(textureMutex);
  if (textures[inputId] != nullptr) {
    paintFrame(textures[inputId]->texture, width() * devicePixelRatio(), height() * devicePixelRatio(), 0, 0);
  }
}
