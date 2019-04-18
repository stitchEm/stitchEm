// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/overlayInputDef.hpp"
#include "deferredUpdater.hpp"

namespace VideoStitch {
namespace Core {

class OverlayInputDefinitionUpdater : public OverlayInputDefinition, public DeferredUpdater<OverlayInputDefinition> {
 public:
  explicit OverlayInputDefinitionUpdater(const OverlayInputDefinition &overlayInputDefinition);

  explicit OverlayInputDefinitionUpdater(OverlayInputDefinition *overlayInputDefinition);

  virtual OverlayInputDefinition *clone() const override;

  virtual Ptv::Value *serialize() const override;

  virtual bool operator==(const OverlayInputDefinition &other) const override;

  virtual bool validate(std::ostream &os) const override;

  virtual const Ptv::Value &getReaderConfig() const override;

  virtual int64_t getWidth() const override;

  virtual int64_t getHeight() const override;

  virtual void setWidth(int64_t int64) override;

  virtual void setHeight(int64_t int64) override;

  virtual frameid_t getFrameOffset() const override;

  virtual void setReaderConfig(Ptv::Value *config) override;

  virtual void setFrameOffset(int fo) override;

  virtual void setFilename(const std::string &fileName) override;

  virtual std::string getDisplayName() const override;

  virtual bool getGlobalOrientationApplied() const override;

  virtual void setGlobalOrientationApplied(const bool status) override;

  virtual const CurveTemplate<double> &getScaleCurve() const override;

  virtual CurveTemplate<double> *displaceScaleCurve(CurveTemplate<double> *newCurve) override;

  virtual void resetScaleCurve() override;

  virtual void replaceScaleCurve(CurveTemplate<double> *newCurve) override;

  virtual const CurveTemplate<double> &getTransXCurve() const override;

  virtual CurveTemplate<double> *displaceTransXCurve(CurveTemplate<double> *newCurve) override;

  virtual void resetTransXCurve() override;

  virtual void replaceTransXCurve(CurveTemplate<double> *newCurve) override;

  virtual const CurveTemplate<double> &getTransYCurve() const override;

  virtual CurveTemplate<double> *displaceTransYCurve(CurveTemplate<double> *newCurve) override;

  virtual void resetTransYCurve() override;

  virtual void replaceTransYCurve(CurveTemplate<double> *newCurve) override;

  virtual const CurveTemplate<double> &getTransZCurve() const override;

  virtual CurveTemplate<double> *displaceTransZCurve(CurveTemplate<double> *newCurve) override;

  virtual void resetTransZCurve() override;

  virtual void replaceTransZCurve(CurveTemplate<double> *newCurve) override;

  virtual const CurveTemplate<double> &getAlphaCurve() const override;

  virtual CurveTemplate<double> *displaceAlphaCurve(CurveTemplate<double> *newCurve) override;

  virtual void resetAlphaCurve() override;

  virtual void replaceAlphaCurve(CurveTemplate<double> *newCurve) override;

  virtual const CurveTemplate<Quaternion<double>> &getRotationCurve() const override;

  virtual CurveTemplate<Quaternion<double>> *displaceRotationCurve(
      CurveTemplate<Quaternion<double>> *newCurve) override;

  virtual void resetRotationCurve() override;

  virtual void replaceRotationCurve(CurveTemplate<Quaternion<double>> *newCurve) override;

 private:
  std::unique_ptr<OverlayInputDefinition> overlayInputDefinition;
};

}  // namespace Core
}  // namespace VideoStitch
