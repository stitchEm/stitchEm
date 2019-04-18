// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PROCESSORSTITCHOUTPUT_HPP_
#define PROCESSORSTITCHOUTPUT_HPP_

#include "config.hpp"
#include "status.hpp"
#include "ptv.hpp"
#include "stitchOutput.hpp"

namespace VideoStitch {
namespace Core {

/**
 * A StitchOutput that applies processing to the output of the stitcher.
 */
class VS_EXPORT ProcessorStitchOutput : public StitchOutput {
 public:
  /**
   * A class that represents predefined processors to apply.
   */
  class Spec {
   public:
    Spec();
    /**
     * Enables computing the sum of solid pixels.
     * "sum": sum_{p in image | p_A > 0}{p_R + p_G + p_B}
     */
    Spec& withSum();
    /**
     * Enables computing the count of solid pixels.
     * "count": count_{p in image | p_A > 0}{1}
     */
    Spec& withCount();

   private:
    friend class ProcessorStitchOutput;
    bool sum;
    bool count;
  };

  /**
   * Creates a ProcessorStitchOutput with size
   * @param w width
   * @param h height
   * @param spec Spec. Must have at least one processor enabled.
   * @note This StitchOutput is NOT thread safe. It is meant to be used by only one stitcher.
   */
  static Potential<ProcessorStitchOutput> create(size_t w, size_t h, const Spec& spec);

  /**
   * Returns the result of running the processing. The contents depend on the applied processors (cf doc).
   * Only valid until the next call to the dirver stitcher (not thread safe).
   */
  virtual const Ptv::Value& getResult() const = 0;

 protected:
  /**
   * Creates form impl @a impl
   */
  explicit ProcessorStitchOutput(Pimpl*);

 private:
  ProcessorStitchOutput(const ProcessorStitchOutput&);
  const ProcessorStitchOutput& operator=(const ProcessorStitchOutput&);
};

}  // namespace Core
}  // namespace VideoStitch

#endif
