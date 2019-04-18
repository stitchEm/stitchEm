// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "../gpu/testing.hpp"

#include <core1/imageMerger.hpp>
#include <core1/imageMapping.hpp>
#include "libvideostitch/imageMergerFactory.hpp"

#include <atomic>

namespace VideoStitch {
namespace Testing {

/**
 * A fake merger.
 */
class FakeImageMerger : public Core::ImageMerger {
 public:
  FakeImageMerger(int imId, const ImageMerger* to, std::atomic<int>* totalNumSetups)
      : Core::ImageMerger(imId, to), totalNumSetups(totalNumSetups) {}

  virtual Status setup(const Core::PanoDefinition&, Core::InputsMap&, const Core::ImageMapping&, GPU::Stream) override {
    if (totalNumSetups) {
      ++*totalNumSetups;
    }
    return Status::OK();
  }

  virtual Status mergeAsync(Core::TextureTarget, const Core::PanoDefinition& /*pano*/,
                            GPU::Buffer<uint32_t> /*panoDevOut*/, GPU::UniqueBuffer<uint32_t>& /*progressivePbo*/,
                            const Core::ImageMapping& /*fromIm*/, bool /**isFirstMerger*/,
                            GPU::Stream /*stream*/) const override {
    return Status::OK();
  }

 private:
  std::atomic<int>* const totalNumSetups;
};

/**
 * A fake reader factory that ignores the given config and creates configurable readers.
 */
class FakeImageMergerFactory : public Core::ImageMergerFactory {
 public:
  /**
   * Creates a fake image merger factory.
   * @param version Core version
   * @param totalNumSetups If not NULL, will acumulate the total numbe rof setups across all created mergers.
   */
  FakeImageMergerFactory(CoreVersion version, std::atomic<int>* totalNumSetups)
      : coreVersion(version), fakeHash("v2_Fake"), totalNumSetups(totalNumSetups) {}

  /**
   * Testing setters
   * @{
   */
  void setHash(const std::string& value) { fakeHash = value; }
  /**
   * @}
   */

  virtual ImageMergerFactory* clone() const { return new FakeImageMergerFactory(*this); }

  virtual Ptv::Value* serialize() const { return NULL; }

  virtual CoreVersion version() const { return coreVersion; }

  virtual Potential<Core::ImageMerger> create(const Core::PanoDefinition& /*pano*/, Core::ImageMapping& fromIm,
                                              const Core::ImageMerger* to, bool) const {
    return new FakeImageMerger(fromIm.getImId(), to, totalNumSetups);
  }

  virtual std::string hash() const { return fakeHash; }

 private:
  const CoreVersion coreVersion;
  std::string fakeHash;
  std::atomic<int>* const totalNumSetups;  // Not owned.
};

}  // namespace Testing
}  // namespace VideoStitch
