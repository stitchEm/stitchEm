// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"
#include "input/inputFrame.hpp"

#include "libvideostitch/input.hpp"
#include "libvideostitch/preprocessor.hpp"
#include "libvideostitch/postprocessor.hpp"
#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/frame.hpp"

#include <vector>

namespace VideoStitch {
namespace Core {

class Buffer;

/**
 * Base class for the implementations of video pipelines.
 */
class VideoPipeline {
 public:
  virtual ~VideoPipeline();

  static Potential<VideoPipeline> createVideoPipeline(const std::vector<Input::VideoReader*>& inputs,
                                                      const std::vector<PreProcessor*>& preprocs = {},
                                                      PostProcessor* postproc = nullptr);

  Status extract(mtime_t date, FrameRate frameRate, std::map<readerid_t, Input::PotentialFrame>& inputBuffers,
                 std::vector<ExtractOutput*>, AlgorithmOutput* algo);
  Status extract(mtime_t date, std::map<readerid_t, Input::PotentialFrame>& inputBuffers, ExtractOutput*);

  /**
   * Preprocessors setter.
   * @note @p ownership is NOT tranferred.
   */
  void setPreProcessors(const std::vector<PreProcessor*>& preprocessors) {
    for (int i = 0; i < (int)preprocessors.size(); ++i) {
      preprocs[(readerid_t)i] = preprocessors[i];
    }
  }

  /**
   * Postprocessor setter.
   * @note @p ownership is NOT tranferred.
   */
  void setPostProcessor(PostProcessor* postprocessor) { postproc = postprocessor; }

 protected:
  VideoPipeline(const std::vector<Input::VideoReader*>&, const std::vector<PreProcessor*>&, PostProcessor*);
  /**
   * Initialize a videoPipeline, trying to allocate memory.
   * Return Status::OK on success, an error on fail
   */
  virtual Status init();

  Status extraction(Input::PotentialFrame inputBuffer, int source, GPU::Surface& readbackDevBuffer, GPU::Stream stream);

  // readers, needed for unpacking
  std::map<readerid_t, Input::VideoReader*> readers;
  // input preprocessors (ownership retained by caller)
  std::map<readerid_t, PreProcessor*> preprocs;
  // output postprocessor (ownership retained by caller)
  PostProcessor* postproc;

  std::map<readerid_t, GPU::Stream> streams;
  std::map<readerid_t, GPU::Buffer<unsigned char>> inputDeviceBuffers;

 private:
  VideoPipeline& operator=(const VideoPipeline&);
};
}  // namespace Core
}  // namespace VideoStitch
