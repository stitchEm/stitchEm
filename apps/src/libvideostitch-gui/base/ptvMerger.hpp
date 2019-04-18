// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "../common.hpp"

#include "libvideostitch/config.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/panoDef.hpp"
#include "version.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>

namespace VideoStitch {
namespace Ptv {
class Value;
}
}  // namespace VideoStitch

namespace VideoStitch {
namespace Helper {

class VS_GUI_EXPORT PtvMerger {
 public:
  /**
   *	Gets the Value obtain by applying a template to currentProject;
   */
  static std::unique_ptr<Ptv::Value> getMergedValue(const std::string &currentPtv, const std::string &templatePtv);
  /**
   *	Apply a template to currentPtv, and saves it on outputPtv. By default outputPtv = currentPtv.
   */
  static void saveMergedPtv(const std::string &currentPtv, const std::string &templatePtv,
                            const std::string &outputPtv);

  static void mergeValue(Ptv::Value *originalValue, Ptv::Value *templateValue);

  static void removeFrom(Ptv::Value *originalValue, Ptv::Value *toRemove);
};

}  // namespace Helper
}  // namespace VideoStitch
