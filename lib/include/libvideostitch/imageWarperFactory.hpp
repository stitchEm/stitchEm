// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef IMAGEWARPERFACTORY_HPP_
#define IMAGEWARPERFACTORY_HPP_

#include "config.hpp"
#include "status.hpp"
#include "object.hpp"

#include <string>
#include <vector>

namespace VideoStitch {
namespace Core {

class ImageWarper;

/**
 * @brief An ImageWarper Factory.
 *
 * VideoStitch provides an ImageWarperFactory for each warper type,
 * but you can add your own by inheriting from ImageWarperFactory.
 */
class VS_EXPORT ImageWarperFactory : public Ptv::Object {
 public:
  virtual ~ImageWarperFactory() {}

  virtual std::string getImageWarperName() const = 0;

  /**
   * Returns the list of available warper.
   */
  static const std::vector<std::string>& availableWarpers();

  /**
   * Returns the list of all compatible wraper to an input flow type.
   */
  static std::vector<std::string> compatibleWarpers(const std::string& flow);

  /**
   * Creates a factory instancefrom a serialized PTV Value.
   * @param value config.
   */
  static Potential<Core::ImageWarperFactory> createWarperFactory(const Ptv::Value* value);

  virtual bool needsInputPreProcessing() const = 0;

  virtual Potential<ImageWarper> create() const = 0;

  virtual ImageWarperFactory* clone() const = 0;

  bool equal(const ImageWarperFactory& other) const;

  static Potential<ImageWarperFactory> newImpotentWarperFactory();

 protected:
 private:
  /**
   * Returns an in-memory hash of the factory. Needs not be backwards-compatible.
   */
  virtual std::string hash() const = 0;
};

}  // namespace Core
}  // namespace VideoStitch

#endif
