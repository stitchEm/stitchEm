// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "controlPointListDef.hpp"
#include "deferredUpdater.hpp"

namespace VideoStitch {
namespace Core {

class VS_EXPORT ControlPointsListUpdater : public ControlPointListDefinition,
                                           public DeferredUpdater<ControlPointListDefinition> {
 public:
  explicit ControlPointsListUpdater(const ControlPointListDefinition &definition);

  virtual ControlPointListDefinition *clone() const override;

  virtual Ptv::Value *serialize() const override;

  virtual bool operator==(const ControlPointListDefinition &other) const override;

  virtual bool validate(std::ostream &os, const videoreaderid_t numVideoInputs) const override;

  virtual const ControlPointList &getCalibrationControlPointList() const override;

  virtual void setCalibrationControlPointList(const ControlPointList &list) override;

 private:
  std::unique_ptr<ControlPointListDefinition> controlPointListDefinition;
};

}  // namespace Core
}  // namespace VideoStitch
