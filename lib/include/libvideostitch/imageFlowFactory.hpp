// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "status.hpp"
#include "object.hpp"

#include <string>
#include <vector>

namespace VideoStitch {
namespace Core {

class ImageFlow;

/**
 * @brief An ImageFlow Factory.
 *
 * VideoStitch provides an ImageFlowFactory for each flow type,
 * but you can add your own by inheriting from ImageFlowFactory.
 */
class VS_EXPORT ImageFlowFactory : public Ptv::Object {
 public:
  virtual ~ImageFlowFactory() {}

  /**
   * Returns the list of available flows.
   */
  static const std::vector<std::string>& availableFlows();

  /**
   * Creates a factory instance from a serialized PTV Value.
   * @param value config.
   */
  static Potential<Core::ImageFlowFactory> createFlowFactory(const Ptv::Value* value);

  virtual Potential<ImageFlow> create() const = 0;

  virtual bool needsInputPreProcessing() const = 0;

  virtual std::string getImageFlowName() const = 0;

  virtual ImageFlowFactory* clone() const = 0;

  bool equal(const ImageFlowFactory& other) const;

  static Potential<ImageFlowFactory> newImpotentFlowFactory();

 protected:
 private:
  /**
   * Returns an in-memory hash of the factory. Needs not be backwards-compatible.
   */
  virtual std::string hash() const = 0;
};

}  // namespace Core
}  // namespace VideoStitch
