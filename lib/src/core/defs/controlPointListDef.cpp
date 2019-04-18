// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panoInputDefsPimpl.hpp"

#include "parse/json.hpp"

#include "libvideostitch/controlPointListDef.hpp"
#include "libvideostitch/parse.hpp"

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <algorithm>

namespace VideoStitch {
namespace Core {

ControlPointListDefinition::Pimpl::Pimpl() {}

ControlPointListDefinition::Pimpl::~Pimpl() {}

ControlPointListDefinition* ControlPointListDefinition::clone() const {
  ControlPointListDefinition* result = new ControlPointListDefinition();

#define AUTO_FIELD_COPY(field) result->set##field(get##field())
#define PIMPL_FIELD_COPY(field) result->pimpl->field = pimpl->field;
  PIMPL_FIELD_COPY(list);
#undef AUTO_FIELD_COPY
#undef PIMPL_FIELD_COPY

  return result;
}

bool ControlPointListDefinition::operator==(const ControlPointListDefinition& other) const {
#define FIELD_EQUAL(getter) (getter() == other.getter())
  if (!(FIELD_EQUAL(getCalibrationControlPointList))) {
    return false;
  }
  return true;
#undef FIELD_EQUAL
}

ControlPointListDefinition::ControlPointListDefinition() : pimpl(new Pimpl()) {}

ControlPointListDefinition::~ControlPointListDefinition() { delete pimpl; }

GENREFGETTER(ControlPointListDefinition, ControlPointList, CalibrationControlPointList, list)
GENREFSETTER(ControlPointListDefinition, ControlPointList, CalibrationControlPointList, list)

bool ControlPointListDefinition::validate(std::ostream& os, videoreaderid_t numVideoInputs) const {
  if (!pimpl->list.empty()) {
    // find maximum input index among index0 and index1 in list to check that the control points are compatible with the
    // number of inputs
    auto result =
        std::max_element(pimpl->list.begin(), pimpl->list.end(), [](const ControlPoint& a, const ControlPoint& b) {
          return std::max(a.index0, a.index1) < std::max(b.index0, b.index1);
        });

    if (!pimpl->list.empty() && std::max(result->index0, result->index1) >= numVideoInputs) {
      os << "the control points reference more inputs (" << std::max(result->index0, result->index1)
         << ") than the current project has (" << numVideoInputs << ")." << std::endl;
      return false;
    }
  }

  return true;
}

Potential<ControlPointListDefinition> ControlPointListDefinition::create(const Ptv::Value& value) {
  std::unique_ptr<ControlPointListDefinition> res(new ControlPointListDefinition());
  FAIL_RETURN(res->applyDiff(value));
  return res.release();
}

Status ControlPointListDefinition::applyDiff(const Ptv::Value& value) {
#define POPULATE_INT_PROPAGATE_WRONGTYPE(config_name, varName)                                           \
  if (Parse::populateInt("ControlPointListDefinition", *c, config_name, varName, true) ==                \
      Parse::PopulateResult_WrongType) {                                                                 \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                                \
            "Invalid type for '" config_name "' in ControlPointListDefinition, expected integer value"}; \
  }
#define POPULATE_DOUBLE_PROPAGATE_WRONGTYPE(config_name, varName)                                       \
  if (Parse::populateDouble("ControlPointListDefinition", *c, config_name, varName, true) ==            \
      Parse::PopulateResult_WrongType) {                                                                \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                               \
            "Invalid type for '" config_name "' in ControlPointListDefinition, expected double value"}; \
  }

  const Ptv::Value* val_calibration_control_points = value.has("calibration_control_points");
  if (val_calibration_control_points && val_calibration_control_points->getType() == Ptv::Value::OBJECT) {
    const Ptv::Value* val_matched_control_points = val_calibration_control_points->has("matched_control_points");
    ControlPointList cpList;
    if (val_matched_control_points && val_matched_control_points->getType() == Ptv::Value::LIST) {
      std::vector<Ptv::Value*> control_points_list = val_matched_control_points->asList();
      for (auto& c : control_points_list) {
        ControlPoint cp;

        POPULATE_INT_PROPAGATE_WRONGTYPE("frame_number", cp.frameNumber);
        POPULATE_INT_PROPAGATE_WRONGTYPE("input_index0", cp.index0);
        POPULATE_INT_PROPAGATE_WRONGTYPE("input_index1", cp.index1);

        POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("x0", cp.x0);
        POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("y0", cp.y0);
        POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("x1", cp.x1);
        POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("y1", cp.y1);
        POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("score", cp.score);

        cpList.push_back(cp);
      }
      setCalibrationControlPointList(cpList);
    }
  }

#undef POPULATE_INT_PROPAGATE_WRONGTYPE
#undef POPULATE_DOUBLE_PROPAGATE_WRONGTYPE

  return Status::OK();
}

Ptv::Value* ControlPointListDefinition::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();

  // sort the list by frameNumber, then index0, then index1, then x0, then y0
  pimpl->list.sort([](const ControlPoint& a, const ControlPoint& b) {
    return (a.frameNumber != b.frameNumber)
               ? (a.frameNumber < b.frameNumber)
               : (a.index0 != b.index0)
                     ? (a.index0 < b.index0)
                     : (a.index1 != b.index1) ? (a.index1 < b.index1) : (a.x0 != b.x0) ? (a.x0 < b.x0) : (a.y0 < b.y0);
  });

  auto cplistData = std::make_unique<std::vector<Ptv::Value*>>();
  for (auto& it : pimpl->list) {
    Ptv::Value* cp = Ptv::Value::emptyObject();

    cp->push("frame_number", new Parse::JsonValue(it.frameNumber));
    cp->push("input_index0", new Parse::JsonValue(int(it.index0)));
    cp->push("x0", new Parse::JsonValue(it.x0));
    cp->push("y0", new Parse::JsonValue(it.y0));
    cp->push("input_index1", new Parse::JsonValue(int(it.index1)));
    cp->push("x1", new Parse::JsonValue(it.x1));
    cp->push("y1", new Parse::JsonValue(it.y1));
    cp->push("score", new Parse::JsonValue(it.score));

    cplistData->push_back(cp);
  }
  if (!cplistData->empty()) {
    res->push("matched_control_points", new Parse::JsonValue(cplistData.release()));
  }

  return res;
}

}  // namespace Core
}  // namespace VideoStitch
