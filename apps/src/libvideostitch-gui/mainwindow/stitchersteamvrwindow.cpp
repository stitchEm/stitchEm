// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "stitchersteamvrwindow.hpp"

const std::string StitcherSteamVRWindow::name = "steamvr";

StitcherSteamVRWindow::StitcherSteamVRWindow(bool isStereo) : SteamVRWindow(isStereo) {}

StitcherSteamVRWindow::~StitcherSteamVRWindow() {
  if (surface) {
    surface->release();
  }
}

std::string StitcherSteamVRWindow::getName() const { return StitcherSteamVRWindow::name; }

void StitcherSteamVRWindow::render(std::shared_ptr<VideoStitch::Core::PanoOpenGLSurface> surf, mtime_t date) {
  if (surface) {
    surface->release();
  }
  surface = surf;
  surf->acquire();
  Texture::getLeft().setPanoramicSize((int)surf->getWidth(), (int)surf->getHeight());
  Texture::getLeft().pixelBuffer = surf->pixelbuffer;
  Texture::getLeft().date = date;
}

void StitcherSteamVRWindow::renderCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface> surf, mtime_t date) {
  _renderCubemap(Texture::Type::CUBEMAP, surf, date);
}

void StitcherSteamVRWindow::renderEquiangularCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface> surf,
                                                     mtime_t date) {
  _renderCubemap(Texture::Type::EQUIANGULAR_CUBEMAP, surf, date);
}

void StitcherSteamVRWindow::_renderCubemap(Texture::Type type,
                                           std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface> surf,
                                           mtime_t date) {
  Texture::getLeft().lock->lock();
  if (surface) {
    surface->release();
  }
  surface = surf;
  surf->acquire();
  Texture::getLeft().setCubemapSize((int)surf->getWidth(), (int)surf->getHeight(), (int)surf->getLength(), type);
  for (int i = 0; i < 6; ++i) {
    Texture::getLeft().pbo[i] = surf->faces[i];
  }
  Texture::getLeft().date = date;
  Texture::getLeft().lock->unlock();
}
