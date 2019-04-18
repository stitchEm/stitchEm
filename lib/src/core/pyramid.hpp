// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/uniqueBuffer.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/logging.hpp"

#include <stdint.h>
#include <vector>

namespace VideoStitch {
namespace Core {

/**
 * @brief A class that computes pyramids of images.
 */
template <typename T>
class LaplacianPyramid {
 public:
  /**
   * @brief The specification of a pyramid level.
   */
  template <typename S>
  class LevelSpec {
   public:
    /**
     * Creates a LevelSpec of size @a width x @a height that holds its data in @a data.
     * @a data must be at least of size @a width * @a height.
     */
    LevelSpec(int64_t width, int64_t height, GPU::Buffer<S> data) : _data(data), _width(width), _height(height) {}
    /**
     * Returns the width of the level.
     */
    int64_t width() const { return _width; }
    /**
     * Returns the height of the level.
     */
    int64_t height() const { return _height; }
    /**
     * Returns the buffer for the level.
     */
    GPU::Buffer<const S> data() const { return _data; }
    /**
     * Returns the buffer for the level.
     */
    GPU::Buffer<S> data() { return _data; }

   private:
    friend class LaplacianPyramid<S>;
    void setDataBuffer(GPU::Buffer<S> dataBuffer) { _data = dataBuffer; }

    void release() { _data.release(); }
    GPU::Buffer<S> _data;
    const int64_t _width;
    const int64_t _height;
  };

  /**
   * Specifies whether the first level should be allocated inside the pyramid or provided by the user.
   * Internal:
   *   pyramid->start(NULL, false, 0);
   *   // [Fill in constBuffer...]
   *   pyramid->compute(constBuffer, stream);
   *   // [Use level contents...]
   *
   *   pyramid->start(NULL, true, 0);
   *   // [Fill level contents...]
   *   pyramid->collapse(outBuffer, stream);
   *
   * External is more efficient and uses less memory, but you need to provide a buffer before calling compute() or
   * collapse(): pyramid->start(buffer, false, 0);
   *   // [Fill in buffer...]
   *   pyramid->compute(stream);
   *   // [Use level contents...]
   *
   *   pyramid->start(buffer, true, 0);
   *   // [Fill level contents...]
   *   pyramid->collapse(stream);
   *   // [Use the buffer...]
   */
  enum LevelLocation { ExternalFirstLevel, InternalFirstLevel };

  /**
   * Specifis whether the reconstruction step should be immutable.
   * If SingleShot, the reconstruction step is destructive and the pyramid must be cleared afterward.
   *
   * If Multiple, double buffering for each level is used and the pyramid need only be cleared to change
   * the input image.
   */
  enum Reconstruction { SingleShot, Multiple };

  /**
   * Create a pyramid.
   * There will be @a numLevels laplacian levels plus one base level.
   * @param width Input/output width
   * @param width Input/output height
   * @param numLevels number of levels
   * @param levelLocation Where the first level resides.
   * @param gaussianRadius The radius of the gaussian blur to go from one level to the other.
   * @param filterPasses The number of filtering filterPasses
   * @param wrap whether the image wraps horizontally.
   */
  static Potential<LaplacianPyramid<T>> create(std::string name, int64_t width, int64_t height, int numLevels,
                                               LevelLocation levelLocation, Reconstruction reconstruction,
                                               int gaussianRadius, int filterPasses, bool wrap);

  /**
   * Start a pyramid session.
   * @param result Mandatory if ExternalFirstLevel : buffer to fill with the reconstruction final result.
   * @param reconstruct Mandatory if Multiple and ExternalFirstLevel : buffer to fill with the various reconstruction
   * results.
   */
  void start(GPU::Buffer<T> result, GPU::Buffer<T> reconstruct, GPU::Stream stream);

  /**
   * Returns the @a level -th LevelSpec of the pyramid.
   */
  const LevelSpec<T>& getLevel(unsigned level) const { return levels[level]; }

  /**
   * Returns the @a level -th LevelSpec of the pyramid.
   */
  LevelSpec<T>& getLevel(unsigned level) { return levels[level]; }

  /**
   * Number of levels, excluding the base level.
   */
  int numLevels() const { return (int)levels.size() - 1; }

  /**
   * Compute the gaussian pyramid of the input buffer.
   * @param stream All computations are done asynchronously there
   *
   */
  Status computeGaussian(GPU::Stream stream);

  /**
   * Compute the laplacian pyramid of the input buffer.
   * @param stream All computations are done asynchronously there
   */
  Status compute(GPU::Stream stream);

  /**
   * Compute the laplacian pyramid of @src. @src should have the correct size.
   * @param src Source buffer.
   * @param stream All computations are done asynchronously there
   * Note that this is less efficient than providing your input buffer in the constructor.
   */
  Status compute(GPU::Buffer<const T> src, GPU::Stream stream);

  /**
   * Collapse the pyramid and write the output in the buffer. This destroys the pyramid data
   * if Reconstruction was not set to Multiple or if final is true.
   *
   * @param stream All computations are done asynchronously there
   */
  Status collapse(bool final, GPU::Stream stream);

  /**
   * Return the size of the pyramid buffer in pixels.
   */
  size_t getBufferSizeInPixels() const { return bufferSizeInPixels; }

  /**
   * Returns true if the image wraps.
   */
  bool wraps() const { return wrap; }

  void setReconstruction(Reconstruction r) { reconstruction = r; }

 protected:
  LaplacianPyramid(std::string name, int64_t bufferSizeInPixels, LevelLocation levelLocation,
                   Reconstruction reconstruction, int gaussianRadius, int filterPasses, bool wrap);

  /**
   * Initializes the pyramid.
   * @param width Input/output width
   * @param height Input/output height
   * @param numLevels number of levels
   */
  Status init(int64_t width, int64_t height, int numLevels);

 private:
  /**
   * Returns the required buffer size for representing a pyramid of
   * size @a width x @a height with @a numLevels levels.
   * This is around 4/3 of the size of the input, but is non-trivial
   * to compute because of non power-of-two sizes.
   * @param width Input buffer width
   * @param height Input buffer height
   * @param nulLevels NUmber of levels.
   */
  static int64_t computeBufferSize(int64_t width, int64_t height, int numLevels);

  std::string name;

  bool wrap;

  GPU::UniqueBuffer<T> pyramid;
  GPU::UniqueBuffer<T> reconstructedPyramid;
  GPU::UniqueBuffer<T> devTmp;
  GPU::UniqueBuffer<T> devTmp2;

  const size_t bufferSizeInPixels;
  const LevelLocation levelLocation;
  Reconstruction reconstruction;
  size_t devBufferSizeInPixels;
  int gaussianRadius;
  int filterPasses;

  std::vector<LevelSpec<T>> levels;
  std::vector<LevelSpec<T>> reconstructedLevels;
};

}  // namespace Core
}  // namespace VideoStitch
