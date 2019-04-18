// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef MERGERMASKUPDATER_HPP
#define MERGERMASKUPDATER_HPP

#include "mergerMaskDef.hpp"
#include "deferredUpdater.hpp"

namespace VideoStitch {
namespace Core {

class MergerMaskUpdater : public MergerMaskDefinition, public DeferredUpdater<MergerMaskDefinition> {
 public:
  explicit MergerMaskUpdater(const MergerMaskDefinition& mergerDefinition);

  virtual MergerMaskDefinition* clone() const override;

  virtual Ptv::Value* serialize() const override;

  virtual bool getEnabled() const override;

  virtual int64_t getWidth() const override;

  virtual int64_t getHeight() const override;

  virtual std::vector<size_t> getMasksOrder() const override;

  virtual int getInputScaleFactor() const override;

  virtual void setEnabled(bool b) override;

  virtual void setWidth(int64_t int641) override;

  virtual void setHeight(int64_t int641) override;

  virtual void setMasksOrder(std::vector<size_t> vector) override;

  virtual void setInputScaleFactor(int) override;

  virtual const InputIndexPixelData& getInputIndexPixelData() const override;

  virtual bool validateInputIndexPixelData() const override;

  virtual Status setInputIndexPixelData(const std::map<videoreaderid_t, std::string>& encodedMasks,
                                        const uint64_t width, const uint64_t height, const frameid_t frameId) override;

  virtual std::vector<frameid_t> getFrameIds() const override;

  virtual void removeFrameIds(const std::vector<frameid_t>& frameIds) override;

  virtual std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>> getInputIndexPixelDataIfValid(
      const frameid_t frameId) const override;

 private:
  std::unique_ptr<MergerMaskDefinition> mergerMaskDefinition;
};

}  // namespace Core
}  // namespace VideoStitch

#endif  // MERGERMASKUPDATER_HPP
