// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/input.hpp"
#include "libvideostitch/ptv.hpp"

namespace VideoStitch {

namespace Util {
class OpaquePtr;
}
class ThreadSafeOstream;

namespace Input {

class Procedure {
 public:
  virtual ~Procedure() {}

  /**
   * Creates a Procedure from a config. Returns NULL on failure.
   */
  static Potential<Procedure> create(const Ptv::Value& config, Util::OpaquePtr** ctx = NULL);

  /**
   * Generate an image in the given buffer.
   * @param frame current frame
   * @param buffer GPU buffer to generate an image into.
   * @param width input width
   * @param height input height
   * @param inputId input id
   */
  virtual void process(frameid_t frame, GPU::Buffer<uint32_t> buffer, int64_t width, int64_t height,
                       readerid_t inputId) const = 0;

  /**
   * Returns a display name for the reader. Never assume any format for that. EVER.
   * That means that the only thing that's allowed with that is to diplay it to the user.
   * No testing for equality, parsing...
   * @param os Output stream.
   */
  virtual void getDisplayName(std::ostream& os) const = 0;

 protected:
  Procedure() {}
};

/**
 * A reader that uses a processor to create a synthetic input.
 */
class ProceduralReader : public VideoReader {
 public:
  static VideoReader* create(readerid_t id, const Ptv::Value& config, int64_t targetWidth, int64_t targetHeight);
  static bool isKnown(const Ptv::Value& config);

  ProceduralReader(readerid_t id, Procedure* processor, int64_t targetWidth, int64_t targetHeight);
  virtual ~ProceduralReader();

  virtual ReadStatus readFrame(mtime_t& date, unsigned char* videoFrame);
  virtual Status seekFrame(frameid_t);

 private:
  static ProceduralReader* createExpressionReader(readerid_t id, const Ptv::Value& config, int64_t targetWidth,
                                                  int64_t targetHeight);
  static ProceduralReader* createFrameNumberReader(readerid_t id, const Ptv::Value& config, int64_t targetWidth,
                                                   int64_t targetHeight);
  static ProceduralReader* createGridReader(readerid_t id, const Ptv::Value& config, int64_t targetWidth,
                                            int64_t targetHeight);

  Procedure* processor;
  mtime_t curDate;
};
}  // namespace Input
}  // namespace VideoStitch
