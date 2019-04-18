// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/controlPointListUpdater.hpp"

namespace VideoStitch {
namespace Core {

ControlPointsListUpdater::ControlPointsListUpdater(const ControlPointListDefinition& definition)
    : controlPointListDefinition(definition.clone()) {}

ControlPointListDefinition* ControlPointsListUpdater::clone() const { return controlPointListDefinition->clone(); }

Ptv::Value* ControlPointsListUpdater::serialize() const { return controlPointListDefinition->serialize(); }

bool ControlPointsListUpdater::operator==(const ControlPointListDefinition& other) const {
  return controlPointListDefinition->operator==(other);
}

bool ControlPointsListUpdater::validate(std::ostream& os, const videoreaderid_t numVideoInputs) const {
  return controlPointListDefinition->validate(os, numVideoInputs);
}

const ControlPointList& ControlPointsListUpdater::getCalibrationControlPointList() const {
  return controlPointListDefinition->getCalibrationControlPointList();
}

void ControlPointsListUpdater::setCalibrationControlPointList(const ControlPointList& list) {
  PRESERVE_ACTION(setCalibrationControlPointList, controlPointListDefinition, list);
}

}  // namespace Core
}  // namespace VideoStitch
