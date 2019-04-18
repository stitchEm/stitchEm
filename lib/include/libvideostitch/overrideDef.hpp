// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "inputDef.hpp"

namespace VideoStitch {
namespace Core {

class OverrideOutputDefinition {
 public:
  bool manualFocal;
  double overrideFocal;

  bool resetRotation;

  bool changeOutputFormat;
  InputDefinition::Format newFormat;

  bool changeOutputSize;
  int64_t width;
  int64_t height;

  void applyOverrideSettings(InputDefinition& inputDef) const;
};

}  // namespace Core
}  // namespace VideoStitch
