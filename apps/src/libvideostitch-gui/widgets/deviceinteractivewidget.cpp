// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "deviceinteractivewidget.hpp"
#include "videostitcher/globalcontroller.hpp"

DeviceInteractiveWidget::DeviceInteractiveWidget(QWidget* parent) : InteractiveWidget(parent), ref_vs(0) {
  clk.start();
  thisPtr = new std::shared_ptr<DeviceInteractiveWidget>(this);
  connect(this, SIGNAL(gotFrame(mtime_t)), this, SLOT(update()));
}

void DeviceInteractiveWidget::syncOn() { sync = true; }

void DeviceInteractiveWidget::syncOff() { sync = false; }

std::string DeviceInteractiveWidget::getName() const { return "interactive view"; }

void DeviceInteractiveWidget::onCloseProject() {
  if (surface) {
    surface->release();
    surface = nullptr;
  }
  releaseMe.clear();
}

void DeviceInteractiveWidget::render(std::shared_ptr<VideoStitch::Core::PanoOpenGLSurface> surf, mtime_t date) {
  Texture::get().lock->lock();
  if (surface) {
    surface->release();
    releaseMe.insert(surface);
  }
  surface = surf;
  surf->acquire();
  Texture::get().setPanoramicSize((int)surf->getWidth(), (int)surf->getHeight());
  Texture::get().pixelBuffer = surf->pixelbuffer;
  Texture::get().date = date;
  Texture::get().lock->unlock();

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

void DeviceInteractiveWidget::renderCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface> surf,
                                            mtime_t date) {
  _renderCubemap(Texture::Type::CUBEMAP, surf, date);
}

void DeviceInteractiveWidget::renderEquiangularCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface> surf,
                                                       mtime_t date) {
  _renderCubemap(Texture::Type::EQUIANGULAR_CUBEMAP, surf, date);
}

void DeviceInteractiveWidget::_renderCubemap(Texture::Type type,
                                             std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface> surf,
                                             mtime_t date) {
  Texture::get().lock->lock();
  if (surface) {
    surface->release();
    releaseMe.insert(surface);
  }
  surface = surf;
  surf->acquire();
  Texture::get().setCubemapSize((int)surf->getWidth(), (int)surf->getHeight(), (int)surf->getLength(), type);
  for (int i = 0; i < 6; ++i) {
    Texture::get().pbo[i] = surf->faces[i];
  }
  Texture::get().date = date;
  Texture::get().lock->unlock();

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

void DeviceInteractiveWidget::registerRenderer(
    std::vector<std::shared_ptr<VideoStitch::Core::PanoRenderer>>* renderers) {
  renderers->push_back(*thisPtr);
}
