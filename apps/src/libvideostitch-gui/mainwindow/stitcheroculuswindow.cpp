// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch-base/logmanager.hpp"

#if defined(Q_OS_WIN)

#include "stitcheroculuswindow.hpp"

#include "libvideostitch-base/oculuswindow.hpp"

const std::string StitcherOculusWindow::name = "oculus";

StitcherOculusWindow::StitcherOculusWindow(bool isStereo, bool mirrorModeEnabled)
    : OculusWindow(isStereo, mirrorModeEnabled) {
  connect(oculus, &OculusRenderer::orientationChanged, this, &StitcherOculusWindow::orientationChanged);
}

StitcherOculusWindow::~StitcherOculusWindow() {
  if (surface) {
    surface->release();
  }
}

bool StitcherOculusWindow::checkOculusCanBeInitialized() {
  if (ovr_Initialize(nullptr) != ovrSuccess) {
    ovrErrorInfo ovrErr;
    ovr_GetLastErrorInfo(&ovrErr);
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(ovrErr.ErrorString);
    QString errMsg = "Failed to initialize the Oculus. Please check that your Oculus is plugged in and switched on.";
    errMsg += " Please also check that the Oculus SDK you are using is > 0.6.0.1";
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(errMsg);
    return false;
  }
  return true;
}

std::string StitcherOculusWindow::getName() const { return StitcherOculusWindow::name; }

void StitcherOculusWindow::render(std::shared_ptr<VideoStitch::Core::PanoOpenGLSurface> surf, mtime_t date) {
  if (surface) {
    surface->release();
  }
  surface = surf;
  surf->acquire();
  //  Texture::getLeft().id = surf->texture;
  Texture::getLeft().setPanoramicSize((int)surf->getWidth(), (int)surf->getHeight());
  Texture::getLeft().pixelBuffer = surf->pixelbuffer;
  Texture::getLeft().date = date;
}

void StitcherOculusWindow::renderCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface> surf, mtime_t date) {
  _renderCubemap(Texture::Type::CUBEMAP, surf, date);
}

void StitcherOculusWindow::renderEquiangularCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface> surf,
                                                    mtime_t date) {
  _renderCubemap(Texture::Type::EQUIANGULAR_CUBEMAP, surf, date);
}

void StitcherOculusWindow::_renderCubemap(Texture::Type type,
                                          std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface> surf, mtime_t date) {
  Texture::getLeft().lock->lock();
  if (surface) {
    surface->release();
  }
  surface = surf;
  surf->acquire();
  Texture::getLeft().setCubemapSize((int)surf->getWidth(), (int)surf->getHeight(), (int)surf->getLength(), type);
  //  Texture::getLeft().id = surf->texture;
  for (int i = 0; i < 6; ++i) {
    Texture::getLeft().pbo[i] = surf->faces[i];
  }
  Texture::getLeft().date = date;
  Texture::getLeft().lock->unlock();
}

#endif
