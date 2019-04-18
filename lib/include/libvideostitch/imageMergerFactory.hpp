// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef IMAGEMERGERFACTORY_HPP_
#define IMAGEMERGERFACTORY_HPP_

#include "config.hpp"
#include "status.hpp"
#include "object.hpp"

#include <string>
#include <vector>

namespace VideoStitch {
namespace Core {

class ImageMerger;
class PanoMerger;
class ImageMapping;
class PanoDefinition;

/**
 * @brief An ImageMerger Factory.
 *
 * VideoStitch provides an ImageMergerFactory for each merger type,
 * but you can add your own by inheriting from ImageMergerFactory.
 */
class VS_EXPORT ImageMergerFactory : public Ptv::Object {
 public:
  virtual ~ImageMergerFactory() {}

  /**
   * Returns the list of available mergers. This is not thread-safe if the lib is compiled in c++ >= 11.
   */
  static const std::vector<std::string>& availableMergers();

  /**
   * Creates a factory instancefrom a serialized PTV Value.
   * @param value config.
   */
  static Potential<Core::ImageMergerFactory> createMergerFactory(const Ptv::Value& value);

  /**
   * Creates a factory instance that cannot instantiate a merger. Useful e.g. for tests.
   */
  static Potential<ImageMergerFactory> newImpotentMergerFactory();

  /**
   * Clones the factory.
   */
  virtual ImageMergerFactory* clone() const = 0;

  // Internal API.

  /**
   * Version identifier.
   */
  enum CoreVersion { CoreVersion1, Depth, Impotent };

  /**
   * Returns the internal merger version.
   */
  virtual CoreVersion version() const = 0;

  /**
   * Returns true if two mergers are the same, including params.
   * @param other Factory to compare against.
   */
  bool equal(const ImageMergerFactory& other) const;

  /**
   * Creates an ImageMerger instance.
   * @param pano The pano definition. Valid only until the function returns, do not take references.
   * @param fromIm The ImageMapping. It's fine to modify the output size.
   * @param to The previous image merger, or NULL if this is the first one. Ownerhip is retained.
   */
  virtual Potential<ImageMerger> create(const PanoDefinition& pano, ImageMapping& fromIm, const ImageMerger* to,
                                        bool progressive) const = 0;

  // TODO fix this design, factory should not have two functions for different core versions..
  virtual Potential<PanoMerger> createDepth(const PanoDefinition& /* pano */) const;

  virtual uint32_t getBlockAlignment() const;

 protected:
  ImageMergerFactory() {}

 private:
  /**
   * Returns an in-memory hash of the factory. Needs not be backwards-compatible.
   */
  virtual std::string hash() const = 0;
};

}  // namespace Core
}  // namespace VideoStitch

#endif
