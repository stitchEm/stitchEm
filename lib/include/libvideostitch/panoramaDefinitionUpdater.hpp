// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PANORAMADEFINITIONUPDATER_HPP
#define PANORAMADEFINITIONUPDATER_HPP

#include "panoDef.hpp"
#include "deferredUpdater.hpp"

namespace VideoStitch {
namespace Core {

class MergerMaskUpdater;
class ControlPointsListUpdater;
class InputDefinitionUpdater;
class OverlayInputDefinitionUpdater;

class VS_EXPORT PanoramaDefinitionUpdater : public PanoDefinition, public DeferredUpdater<PanoDefinition> {
 public:
  explicit PanoramaDefinitionUpdater(const PanoDefinition &panoDefinition);
  PanoramaDefinitionUpdater(PanoramaDefinitionUpdater &&rhs);

  virtual bool validate(std::ostream &os) const override;

  virtual bool validateInputMasks() const override;

  virtual const MergerMaskDefinition &getMergerMask() const override;

  virtual MergerMaskDefinition &getMergerMask() override;

  virtual bool getBlendingMaskEnabled() const override;

  virtual void setBlendingMaskEnabled(const bool enabled) override;

  virtual bool getBlendingMaskInterpolationEnabled() const override;

  virtual void setBlendingMaskInterpolationEnabled(const bool enabled) override;

  virtual int64_t getBlendingMaskWidth() const override;

  virtual int64_t getBlendingMaskHeight() const override;

  virtual std::vector<size_t> getMasksOrder() const override;

  virtual int getBlendingMaskInputScaleFactor() const override;

  virtual std::pair<frameid_t, frameid_t> getBlendingMaskBoundedFrameIds(const frameid_t frameId) const override;

  virtual void removeBlendingMaskFrameIds(const std::vector<frameid_t> &frameIds) override;

  virtual std::vector<frameid_t> getBlendingMaskFrameIds() const override;

  virtual std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>> getInputIndexPixelDataIfValid(
      const frameid_t frameId) const override;

  virtual const InputDefinition &getInput(readerid_t i) const override;

  virtual InputDefinition &getInput(readerid_t i) override;

  virtual void insertInput(InputDefinition *inputDef, readerid_t i) override;

  virtual InputDefinition *popInput(readerid_t i) override;

  virtual bool removeInput(int i) override;

  virtual readerid_t numInputs() const override;

  virtual const OverlayInputDefinition &getOverlay(overlayreaderid_t i) const override;

  virtual OverlayInputDefinition &getOverlay(overlayreaderid_t i) override;

  virtual void insertOverlay(OverlayInputDefinition *overlayDef, overlayreaderid_t i) override;

  virtual OverlayInputDefinition *popOverlay(overlayreaderid_t i) override;

  virtual bool removeOverlay(overlayreaderid_t i) override;

  virtual overlayreaderid_t numOverlays() const override;

  virtual const ControlPointListDefinition &getControlPointListDef() const override;

  virtual ControlPointListDefinition &getControlPointListDef() override;

  virtual bool getPrecomputedCoordinateBuffer() const override;

  virtual void setPrecomputedCoordinateBuffer(const bool b) override;

  virtual double getPrecomputedCoordinateShrinkFactor() const override;

  virtual void setPrecomputedCoordinateShrinkFactor(const double b) override;

  virtual int64_t getWidth() const override;

  virtual int64_t getHeight() const override;

  virtual int64_t getLength() const override;

  virtual const Ptv::Value *getPostprocessors() const override;

  virtual double getHFOV() const override;

  virtual double getVFOV() const override;

  virtual double getSphereScale() const override;

  virtual void setSphereScale(double scale) override;

  virtual void setCalibrationCost(double cost) override;

  virtual double getCalibrationCost() const override;

  virtual void setCalibrationInitialHFOV(double hfov) override;

  virtual double getCalibrationInitialHFOV() const override;

  virtual void setHasBeenCalibrationDeshuffled(const bool deshuffled) override;

  virtual bool hasBeenCalibrationDeshufled() const override;

  virtual void setCalibrationControlPointList(const ControlPointList &list) override;

  virtual const ControlPointList &getCalibrationControlPointList() const override;

  virtual const RigDefinition &getCalibrationRigPresets() const override;

  virtual void setCalibrationRigPresets(RigDefinition *rigDef) override;

  virtual std::string getCalibrationRigPresetsName() const override;

  virtual bool hasCalibrationRigPresets() const override;

  virtual void setWidth(uint64_t w) override;

  virtual void setHeight(uint64_t h) override;

  virtual void setLength(uint64_t) override;

  virtual const CurveTemplate<double> &getRedCB() const override;

  virtual CurveTemplate<double> *displaceRedCB(CurveTemplate<double> *newCurve) override;

  virtual void resetRedCB() override;

  virtual void replaceRedCB(CurveTemplate<double> *newCurve) override;

  virtual void resetGreenCB() override;

  virtual const CurveTemplate<double> &getGreenCB() const override;

  virtual void replaceGreenCB(CurveTemplate<double> *newCurve) override;

  virtual CurveTemplate<double> *displaceGreenCB(CurveTemplate<double> *newCurve) override;

  virtual CurveTemplate<double> *displaceBlueCB(CurveTemplate<double> *newCurve) override;

  virtual void resetBlueCB() override;

  virtual const CurveTemplate<double> &getBlueCB() const override;

  virtual void replaceBlueCB(CurveTemplate<double> *newCurve) override;

  virtual const CurveTemplate<double> &getExposureValue() const override;

  virtual void resetExposureValue() override;

  virtual CurveTemplate<double> *displaceExposureValue(CurveTemplate<double> *newCurve) override;

  virtual void replaceExposureValue(CurveTemplate<double> *newCurve) override;

  virtual void setProjection(PanoProjection format) override;

  virtual void setHFOV(double hFov) override;

  virtual void setVFOV(double vFov) override;

  virtual void resetGlobalOrientation() override;

  virtual const CurveTemplate<Quaternion<double>> &getGlobalOrientation() const override;

  virtual CurveTemplate<Quaternion<double>> *displaceGlobalOrientation(
      CurveTemplate<Quaternion<double>> *newCurve) override;

  virtual void replaceGlobalOrientation(CurveTemplate<Quaternion<double>> *newCurve) override;

  virtual CurveTemplate<Quaternion<double>> *displaceStabilization(
      CurveTemplate<Quaternion<double>> *newCurve) override;

  virtual void resetStabilization() override;

  virtual const CurveTemplate<Quaternion<double>> &getStabilization() const override;

  virtual void replaceStabilization(CurveTemplate<Quaternion<double>> *newCurve) override;

  virtual CurveTemplate<double> *displaceStabilizationYaw(CurveTemplate<double> *newCurve) override;

  virtual void replaceStabilizationYaw(CurveTemplate<double> *newCurve) override;

  virtual const CurveTemplate<double> &getStabilizationYaw() const override;

  virtual void resetStabilizationYaw() override;

  virtual void replaceStabilizationPitch(CurveTemplate<double> *newCurve) override;

  virtual void resetStabilizationPitch() override;

  virtual const CurveTemplate<double> &getStabilizationPitch() const override;

  virtual CurveTemplate<double> *displaceStabilizationPitch(CurveTemplate<double> *newCurve) override;

  virtual const CurveTemplate<double> &getStabilizationRoll() const override;

  virtual void resetStabilizationRoll() override;

  virtual CurveTemplate<double> *displaceStabilizationRoll(CurveTemplate<double> *newCurve) override;

  virtual void replaceStabilizationRoll(CurveTemplate<double> *newCurve) override;

  virtual void computeOptimalPanoSize(unsigned &width, unsigned &height) const override;

  virtual ~PanoramaDefinitionUpdater();

  virtual void apply(PanoDefinition &updateValue) override;

 private:
  void initInputUpdaters();
  std::vector<std::shared_ptr<InputDefinitionUpdater>> inputUpdaters;
  bool inputsManaged = false;

  void initOverlayUpdaters();
  std::vector<std::shared_ptr<OverlayInputDefinitionUpdater>> overlayUpdaters;
  bool overlaysManaged = false;

  std::unique_ptr<PanoDefinition> panoramaDefinition;

  std::unique_ptr<MergerMaskUpdater> mergerMaskUpdater;
  std::unique_ptr<ControlPointsListUpdater> controlPointsListUpdater;
};

}  // namespace Core
}  // namespace VideoStitch

#endif  // PANORAMADEFINITIONUPDATER_HPP
