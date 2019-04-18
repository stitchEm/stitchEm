// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "status.hpp"
#include "ptv.hpp"
#include "frame.hpp"
#include "gpu_device.hpp"

#include "gpu_device.hpp"

#include <string>
#include <utility>
#include <vector>

namespace VideoStitch {

namespace Core {
class PanoDefinition;
}

namespace GPU {
class Surface;
}

namespace Util {

/**
 * An opaque pointer conveying state for the algorithm when applicable.
 * Is meant to be used for memoization.
 */
class VS_EXPORT OpaquePtr {
 public:
  virtual ~OpaquePtr() {}
};

/**
 * An interface for a generic algorithm on a PanoDefinition.
 * Algorithms are NOT thread-safe.
 */
class VS_EXPORT Algorithm {
 public:
  /**
   * A class used to report progress. The typical use case is when launching an algorithm in a separate thread.
   * In that case, a thread-safe ProgressReporter is provided that resides in the launcher thread and display updates so
   * that the user does not fall asleep.
   */
  class ProgressReporter {
   public:
    virtual ~ProgressReporter() {}

    /**
     * Called at each checkpoint. Make sure not to block, because the algorithm will wait before resuming execution.
     * @param message Progress message.
     * @param percent Estimate of the progress in percents.
     * Return true to cancel the algorithm (may not be supported by all algorithms).
     */
    virtual bool notify(const std::string& message, double percent) = 0;
  };

  /**
   * Returns the list of available algorithms.
   */
  static void list(std::vector<std::string>& algos);

  /**
   * Returns the default config / doc string for the given algorithm.
   * @param name Name of the algorithm.
   */
  static const char* getDocString(const std::string& name);

  /**
   * Instantiates an algorithm by name. See list() for a list of algorithms and getDocString() for their documentation.
   * @param name Name of the algorithm.
   * @param config Algorithm configuration. Any missing (or incorrectly typed) entry will use the default value. nullptr
   * means use all defaults.
   * @return The algorithm, or nullptr if no such algorithm exists.
   */
  static Potential<Algorithm> create(const std::string& name, const Ptv::Value* config = nullptr);

  virtual ~Algorithm();

  /**
   * Applies the algorithm on a PanoDefinition.
   * @param pano the PanoDefinition
   * @param progress if non-nullptr, used as progress indicator and early stop controller.
   * @param ctx Internal opaque pointer, reserved.
   * @return A status, and possibly with additional values (depending on the exact algorithm and configuration).
   */
  virtual Potential<Ptv::Value> apply(Core::PanoDefinition* pano, ProgressReporter* progress = nullptr,
                                      OpaquePtr** ctx = nullptr) const = 0;

 protected:
  Algorithm();
};

/**
 * An interface for a generic algorithm on a PanoDefinition.
 * This class represents algorithms that are online, meaning they are applied
 * while the stitcher is running by directly accessing its buffers.
 * The callback function is used by the stitcher once the buffers are ready.
 */
class VS_EXPORT OnlineAlgorithm {
 public:
  /**
   * Returns the list of available algorithms.
   * @param algos Return vector.
   */
  static void list(std::vector<std::string>& algos);

  /**
   * Returns the default config / doc string for the given algorithm.
   * @param name Name of the algorithm.
   */
  static const char* getDocString(const std::string& name);

  /**
   * Instantiates an algorithm by name. See list() for a list of algorithms and getDocString() for their documentation.
   * @param name Name of the algorithm.
   * @param config Algorithm configuration. Any missing (or incorrectly typed) entry will use the default value. nullptr
   * means use all defaults.
   * @return The algorithm, or nullptr if no such algorithm exists.
   */
  static Potential<OnlineAlgorithm> create(const std::string& name, const Ptv::Value* config = nullptr);

  virtual ~OnlineAlgorithm();

  /**
   * Applies the algorithm on a PanoDefinition.
   * @param pano the PanoDefinition
   * @param inputFrames
   * @param ctx Internal opaque pointer. Reserved.
   * @return A status, and possibly with additional values (depending on the exact algorithm and configuration).
   */
  virtual Potential<Ptv::Value> onFrame(Core::PanoDefinition& pano,
                                        std::vector<std::pair<videoreaderid_t, GPU::Surface&>>& inputFrames,
                                        mtime_t date = 0, FrameRate frameRate = FrameRate(),
                                        Util::OpaquePtr** ctx = nullptr) = 0;

 protected:
  OnlineAlgorithm();
};
}  // namespace Util
}  // namespace VideoStitch
