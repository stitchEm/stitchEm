// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/overrideDef.hpp"

#include "core/transformGeoParams.hpp"

namespace VideoStitch {
namespace Core {

void OverrideOutputDefinition::applyOverrideSettings(InputDefinition& inputDef) const {
  const GeometryDefinition origGeometry = inputDef.getGeometries().at(0);
  GeometryDefinition geometry = origGeometry;

  if (resetRotation && (origGeometry.getYaw() != 0. || origGeometry.getPitch() != 0. || origGeometry.getRoll() != 0.)) {
    geometry.setRoll(0);
    geometry.setYaw(0);
    geometry.setPitch(0);

    // fixup translation after changing the camera reference:
    // t' = R^t * t
    //
    // `R^t * t` is already computed in the inverse pose (T^-1 = [R^t | -R^t * t]),
    // we can reuse the (negated) value from there
    const TransformGeoParams tgp(inputDef, origGeometry, 1.0);
    const vsfloat3x4 inversePose = tgp.getPoseInverse();

    geometry.setTranslationX(-inversePose.values[0][3]);
    geometry.setTranslationY(-inversePose.values[1][3]);
    geometry.setTranslationZ(-inversePose.values[2][3]);
  }

  if (changeOutputFormat) {
    inputDef.setFormat(newFormat);
  }

  if (changeOutputSize) {
    inputDef.setWidth(width);
    inputDef.setHeight(height);
  }

  const float outputFocal = static_cast<float>(manualFocal ? overrideFocal : inputDef.computeFocalWithoutDistortion());

  geometry.setHorizontalFocal(outputFocal);
  geometry.setVerticalFocal(outputFocal);

  inputDef.replaceGeometries(new GeometryDefinitionCurve(geometry));
}

}  // namespace Core
}  // namespace VideoStitch
