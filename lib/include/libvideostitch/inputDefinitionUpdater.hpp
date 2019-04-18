// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "inputDef.hpp"
#include "deferredUpdater.hpp"

namespace VideoStitch {
namespace Core {

class InputDefinitionUpdater : public InputDefinition, public DeferredUpdater<InputDefinition> {
 public:
  explicit InputDefinitionUpdater(const InputDefinition &inputDefinition);

  // Acquires ownership
  explicit InputDefinitionUpdater(InputDefinition *inputDefinition);

  virtual InputDefinition *clone() const override;

  virtual Ptv::Value *serialize() const override;

  virtual bool operator==(const InputDefinition &other) const override;

  virtual bool validate(std::ostream &os) const override;

  virtual std::string getDisplayName() const override;

  virtual const Ptv::Value &getReaderConfig() const override;

  virtual const std::string &getMaskData() const override;

  virtual bool deletesMaskedPixels() const override;

  virtual const MaskPixelData &getMaskPixelData() const override;

  virtual const unsigned char *getMaskPixelDataIfValid() const override;

  virtual bool validateMask() const override;

  virtual group_t getGroup() const override;

  virtual void setGroup(InputDefinition::group_t group) override;

  virtual int64_t getWidth() const override;

  virtual int64_t getHeight() const override;

  virtual void setWidth(int64_t int64) override;

  virtual void setHeight(int64_t int64) override;

  virtual int64_t getCroppedWidth() const override;

  virtual int64_t getCroppedHeight() const override;

  virtual bool getUseMeterDistortion() const override;

  virtual Format getFormat() const override;

  virtual void setFormat(Format) override;

  virtual LensModelCategory getLensModelCategory() const override;

  virtual bool hasCroppedArea() const override;

  virtual frameid_t getFrameOffset() const override;

  virtual double getSynchroCost() const override;

  virtual int getStack() const override;

  virtual void setStack(int value) override;

  virtual void setFilename(const std::string &fileName) override;

  virtual void setReaderConfig(Ptv::Value *config) override;

  virtual void setMaskData(const std::string &maskData) override;

  virtual void setDeletesMaskedPixels(bool value) override;

  virtual bool setMaskPixelData(const char *buffer, uint64_t maskWidth, uint64_t maskHeight) override;

  virtual void setFrameOffset(int fo) override;

  virtual void setSynchroCost(double cost) override;

  virtual void resetRedCB() override;

  virtual CurveTemplate<double> *displaceRedCB(CurveTemplate<double> *newCurve) override;

  virtual const CurveTemplate<double> &getRedCB() const override;

  virtual void replaceRedCB(CurveTemplate<double> *newCurve) override;

  virtual CurveTemplate<double> *displaceGreenCB(CurveTemplate<double> *newCurve) override;

  virtual void resetGreenCB() override;

  virtual const CurveTemplate<double> &getGreenCB() const override;

  virtual void replaceGreenCB(CurveTemplate<double> *newCurve) override;

  virtual void resetBlueCB() override;

  virtual void replaceBlueCB(CurveTemplate<double> *newCurve) override;

  virtual CurveTemplate<double> *displaceBlueCB(CurveTemplate<double> *newCurve) override;

  virtual const CurveTemplate<double> &getBlueCB() const override;

  virtual void resetExposureValue() override;

  virtual void replaceExposureValue(CurveTemplate<double> *newCurve) override;

  virtual CurveTemplate<double> *displaceExposureValue(CurveTemplate<double> *newCurve) override;

  virtual const CurveTemplate<double> &getExposureValue() const override;

  virtual CurveTemplate<GeometryDefinition> *displaceGeometries(CurveTemplate<GeometryDefinition> *newCurve) override;

  virtual void replaceGeometries(CurveTemplate<GeometryDefinition> *newCurve) override;

  virtual const CurveTemplate<GeometryDefinition> &getGeometries() const override;

  virtual void resetGeometries(const double HFOV) override;

  virtual const Ptv::Value *getPreprocessors() const override;

  virtual void setIsEnabled(bool state) override;

  virtual bool getIsEnabled() const override;

  virtual bool getIsVideoEnabled() const override;

  virtual bool getIsAudioEnabled() const override;

  virtual void setUseMeterDistortion(bool meter) override;

  virtual PhotoResponse getPhotoResponse() const override;

  virtual double getEmorA() const override;

  virtual double getEmorB() const override;

  virtual double getEmorC() const override;

  virtual double getEmorD() const override;

  virtual double getEmorE() const override;

  virtual double getGamma() const override;

  virtual void setEmorA(double emorA) override;

  virtual void setEmorB(double emorB) override;

  virtual void setEmorC(double emorC) override;

  virtual void setEmorD(double emorD) override;

  virtual void setEmorE(double emorE) override;

  virtual void setEmorPhotoResponse(double emorA, double emorB, double emorC, double emorD, double emorE) override;

  virtual void resetPhotoResponse() override;

  virtual void setGamma(double gamma) override;

  virtual double getVignettingCoeff0() const override;

  virtual double getVignettingCoeff1() const override;

  virtual double getVignettingCoeff2() const override;

  virtual double getVignettingCoeff3() const override;

  virtual double getVignettingCenterX() const override;

  virtual double getVignettingCenterY() const override;

  virtual void setVignettingCoeff0(double vignettingCoeff0) override;

  virtual void setVignettingCoeff1(double vignettingCoeff1) override;

  virtual void setVignettingCoeff2(double vignettingCoeff2) override;

  virtual void setVignettingCoeff3(double vignettingCoeff3) override;

  virtual void setVignettingCenterX(double vignettingCenterX) override;

  virtual void setVignettingCenterY(double vignettingCenterY) override;

  virtual void setRadialVignetting(double vignettingCoeff0, double vignettingCoeff1, double vignettingCoeff2,
                                   double vignettingCoeff3, double vignettingCenterX,
                                   double vignettingCenterY) override;

  virtual void resetVignetting() override;

  virtual double getInputCenterX() const override;

  virtual double getInputCenterY() const override;

  virtual double getCenterX(const GeometryDefinition &geometry) const override;

  virtual double getCenterY(const GeometryDefinition &geometry) const override;

  virtual int64_t getCropLeft() const override;

  virtual int64_t getCropRight() const override;

  virtual int64_t getCropTop() const override;

  virtual int64_t getCropBottom() const override;

  virtual void setCropLeft(int64_t left) override;

  virtual void setCropRight(int64_t right) override;

  virtual void setCropTop(int64_t top) override;

  virtual void setCropBottom(int64_t bottom) override;

  virtual void setCrop(int64_t left, int64_t right, int64_t top, int64_t bottom) override;

 private:
  std::unique_ptr<InputDefinition> inputDefinition;
};

}  // namespace Core
}  // namespace VideoStitch
