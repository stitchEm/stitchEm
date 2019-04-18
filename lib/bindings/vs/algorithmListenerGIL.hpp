// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "../../include/libvideostitch/stitchOutput.hpp"
#include "../../include/libvideostitch/geometryDef.hpp"
#include "../../include/libvideostitch/panoDef.hpp"
#include "../../include/libvideostitch/inputDef.hpp"
#include "../../include/libvideostitch/curves.hpp"
#include "../../include/libvideostitch/panoramaDefinitionUpdater.hpp"
#include <memory>

struct AlgorithmListenerGIL : public VideoStitch::Core::AlgorithmOutput::Listener {
 public:
  AlgorithmListenerGIL(VideoStitch::Core::AlgorithmOutput::Listener* ilistener) : internal_listener(ilistener) {}

  virtual void onPanorama(VideoStitch::Core::PanoramaDefinitionUpdater& pano) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    internal_listener->onPanorama(pano);

    /* Release the thread. No Python API allowed beyond this point. */
    PyGILState_Release(gstate);
  }

  virtual void onError(const VideoStitch::Status& status) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    internal_listener->onError(status);

    PyGILState_Release(gstate);
  }

 private:
  VideoStitch::Core::AlgorithmOutput::Listener* internal_listener;
};

VideoStitch::Core::AlgorithmOutput::Listener* toListener(AlgorithmListenerGIL* originalPtr) {
  // swig makes me cry sometimes
  return originalPtr;
}
