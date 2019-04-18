/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * <b>NVIDIA Multimedia API: Video Decode API</b>
 *
 */

/**
 * @defgroup ee_nvvideodecoder_group Video Decoder
 * @ingroup ee_nvvideo_group
 *
 * Helper class that creates new V4L2
 * video decoders, and it sets decoder capture and output plane
 * formats.
 * @{
 */

#ifndef __NV_VIDEO_DECODER_H__
#define __NV_VIDEO_DECODER_H__

#include "NvV4l2Element.h"

/**
 * @brief Defines a helper class for V4L2 Video Decoder.
 *
 * The video decoder device node is `/dev/nvhost-nvdec`. The category name
 * for encoder is \c "NVDEC".
 *
 * Refer to [V4L2 Video Decoder](group__V4L2Dec.html) for more information on the converter.
 */
class NvVideoDecoder : public NvV4l2Element {
 public:
  /**
   * Creates a new V4L2 Video Decoder object named \a name.
   *
   * This method internally calls \c v4l2_open on the decoder dev node
   * \c "/dev/nvhost-nvdec" and checks for \c V4L2_CAP_VIDEO_M2M_MPLANE
   * capability on the device. This method allows the caller to specify
   * additional flags with which the device should be opened.
   *
   * The device is opened in blocking mode, which can be modified by passing
   * the @a O_NONBLOCK flag to this method.
   *
   * @returns Reference to the newly created decoder object else \a NULL in
   *          case of failure during initialization.
   */
  static NvVideoDecoder *createVideoDecoder(const char *name, int flags = 0);

  ~NvVideoDecoder();
  /**
   * Sets the format on the decoder output plane.
   *
   * Calls \c VIDIOC_S_FMT IOCTL internally on the capture plane.
   *
   * @param[in] pixfmt One of the raw V4L2 pixel formats.
   * @param[in] width Width of the output buffers in pixels.
   * @param[in] height Height of the output buffers in pixels.
   * @return 0 for success, -1 otherwise.
   */
  int setCapturePlaneFormat(uint32_t pixfmt, uint32_t width, uint32_t height);
  /**
   * Sets the format on the decoder output plane.
   *
   * Calls the \c VIDIOC_S_FMT IOCTL internally on the output plane.
   *
   * @param[in] pixfmt One of the coded V4L2 pixel formats.
   * @param[in] sizeimage Maximum size of the buffers on the output plane.
                          containing encoded data in bytes.
   * @return 0 for success, -1 otherwise.
   */
  int setOutputPlaneFormat(uint32_t pixfmt, uint32_t sizeimage);

  /**
   * Informs the decoder that input buffers may not contain complete frames.
   *
   * Calls the VIDIOC_S_EXT_CTRLS IOCTL internally with Control ID
   * V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT. Must be called before
   * setFormat on any of the planes.
   *
   * @return 0 for success, -1 otherwise.
   */
  int disableCompleteFrameInputBuffer();

  /**
   * Disables the display picture buffer.
   *
   * Calls the VIDIOC_S_EXT_CTRLS IOCTL internally with Control ID
   * V4L2_CID_MPEG_VIDEO_DISABLE_DPB. Must be called after setFormat on both
   * the planes and before requestBuffers on any of the planes.
   *
   * @return 0 for success, -1 otherwise.
   */
  int disableDPB();

  /**
   * Gets the minimum number of buffers to be requested on the decoder capture plane.
   *
   * Calls the VIDIOC_G_CTRL IOCTL internally with Control ID
   * V4L2_CID_MIN_BUFFERS_FOR_CAPTURE. It is valid after the first
   * V4L2_RESOLUTION_CHANGE_EVENT and may change after each subsequent
   * event.
   *
   * @param[out] num A reference to the integer to return the number of buffers.
   *
   * @return 0 for success, -1 otherwise.
   */
  int getMinimumCapturePlaneBuffers(int &num);

  /**
   * Sets the skip-frames parameter of the decoder.
   *
   * Calls the VIDIOC_S_EXT_CTRLS IOCTL internally with Control ID
   * V4L2_CID_MPEG_VIDEO_SKIP_FRAMES. Must be called after setFormat on both
   * the planes and before requestBuffers on any of the planes.
   *
   * @param[in] skip_frames Type of frames to skip decoding, one of
   *                        enum v4l2_skip_frames_type.
   *
   * @return 0 for success, -1 otherwise.
   */
  int setSkipFrames(enum v4l2_skip_frames_type skip_frames);

  /**
   * Enables video decoder output metadata reporting.
   *
   * Calls the VIDIOC_S_EXT_CTRLS IOCTL internally with Control ID
   * V4L2_CID_MPEG_VIDEO_ERROR_REPORTING. Must be called after setFormat on
   * both the planes and before requestBuffers on any of the planes.
   *
   * @return 0 for success, -1 otherwise.
   */
  int enableMetadataReporting();

  /**
   * Gets metadata for the decoded capture plane buffer.
   *
   * Calls the VIDIOC_S_EXT_CTRLS IOCTL internally with Control ID
   * V4L2_CID_MPEG_VIDEODEC_METADATA. Must be called for a buffer that has
   * been dequeued from the capture plane. The returned metadata corresponds
   * to the last dequeued buffer with index @a buffer_index.
   *
   * @param[in] buffer_index Index of the capture plane buffer whose metadata
   *              is required.
   * @param[in,out] metadata Reference to the metadata structure
   *              v4l2_ctrl_videodec_outputbuf_metadata to be filled.
   *
   * @return 0 for success, -1 otherwise.
   */
  int getMetadata(uint32_t buffer_index, v4l2_ctrl_videodec_outputbuf_metadata &metadata);

 private:
  /**
   * Constructor used by #createVideoDecoder.
   */
  NvVideoDecoder(const char *name, int flags);

  static const NvElementProfiler::ProfilerField valid_fields =
      NvElementProfiler::PROFILER_FIELD_TOTAL_UNITS | NvElementProfiler::PROFILER_FIELD_FPS;
};
/** @} */
#endif
