/*
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  1. Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *  3. The names of its contributors may not be used to endorse or promote
 *     products derived from this software without specific prior written
 *     permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 *  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 *  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This file contains amendments to the V4L2 headers made after the
 * supported kernel version and NVIDIA extensions.
 */

#ifndef __V4L2_NV_EXTENSIONS_H__
#define __V4L2_NV_EXTENSIONS_H__

/**
 * @file
 * <b>NVIDIA V4L2 API Extensions</b>
 *
 * @b Description: This file declares the NVIDIA V4L2 Extensions,
 *  controls and structures.
 */

/**
 * V4L2 pixel format for H265
 */
#define V4L2_PIX_FMT_H265 v4l2_fourcc('H', '2', '6', '5')

/** @cond UNUSED */

#define V4L2_PIX_FMT_YUV444M v4l2_fourcc('Y', 'M', '4', '4')
#define V4L2_PIX_FMT_YUV422M v4l2_fourcc('4', '2', '2', 'M')
#define V4L2_PIX_FMT_YUV422RM v4l2_fourcc('4', '2', 'R', 'M')

#define V4L2_PIX_FMT_H264_SLICE v4l2_fourcc('S', '2', '6', '4') /* H264 parsed slices */
#define V4L2_PIX_FMT_VP8_FRAME v4l2_fourcc('V', 'P', '8', 'F')  /* VP8 parsed frames */

#define V4L2_CTRL_FLAG_CAN_STORE 0x0200

/** @endcond */

/**
 * V4L2 event type for decoder resolution event change
 */
#define V4L2_EVENT_RESOLUTION_CHANGE 5

/** @cond UNUSED */

/*---------------Below are changes from the v4l2-controls.h----------------------*/

#define V4L2_CID_MPEG_VIDEO_H264_SPS (V4L2_CID_MPEG_BASE + 383)
#define V4L2_CID_MPEG_VIDEO_H264_PPS (V4L2_CID_MPEG_BASE + 384)
#define V4L2_CID_MPEG_VIDEO_H264_SCALING_MATRIX (V4L2_CID_MPEG_BASE + 385)
#define V4L2_CID_MPEG_VIDEO_H264_SLICE_PARAM (V4L2_CID_MPEG_BASE + 386)
#define V4L2_CID_MPEG_VIDEO_H264_DECODE_PARAM (V4L2_CID_MPEG_BASE + 387)

#define V4L2_CID_MPEG_VIDEO_VP8_FRAME_HDR (V4L2_CID_MPEG_BASE + 512)

/** @endcond */

/**
 * Control ID to set the H.265 encoder profile.
 *
 * One of #v4l2_mpeg_video_h265_profile should be passed.
 */
#define V4L2_CID_MPEG_VIDEO_H265_PROFILE (V4L2_CID_MPEG_BASE + 513)

/**
 * Enum containing the possible profiles for H.265 encoder.
 */
enum v4l2_mpeg_video_h265_profile {
  /** H.265 Main profile */
  V4L2_MPEG_VIDEO_H265_PROFILE_MAIN = 0,
  /** H.265 Main10 profile */
  V4L2_MPEG_VIDEO_H265_PROFILE_MAIN10 = 1,
  /** H.265 MainStillPicture profile */
  V4L2_MPEG_VIDEO_H265_PROFILE_MAINSTILLPICTURE = 2,
};

/**
 * Control ID to set the encoder IDR frame interval.
 * Should be used with VIDIOC_S_EXT_CTRLS ioctl.
 */
#define V4L2_CID_MPEG_VIDEO_IDR_INTERVAL (V4L2_CID_MPEG_BASE + 514)

/** @cond UNUSED */

/* Complex controls */

#define V4L2_H264_SPS_CONSTRAINT_SET0_FLAG 0x01
#define V4L2_H264_SPS_CONSTRAINT_SET1_FLAG 0x02
#define V4L2_H264_SPS_CONSTRAINT_SET2_FLAG 0x04
#define V4L2_H264_SPS_CONSTRAINT_SET3_FLAG 0x08
#define V4L2_H264_SPS_CONSTRAINT_SET4_FLAG 0x10
#define V4L2_H264_SPS_CONSTRAINT_SET5_FLAG 0x20

#define V4L2_H264_SPS_FLAG_SEPARATE_COLOUR_PLANE 0x01
#define V4L2_H264_SPS_FLAG_QPPRIME_Y_ZERO_TRANSFORM_BYPASS 0x02
#define V4L2_H264_SPS_FLAG_DELTA_PIC_ORDER_ALWAYS_ZERO 0x04
#define V4L2_H264_SPS_FLAG_GAPS_IN_FRAME_NUM_VALUE_ALLOWED 0x08
#define V4L2_H264_SPS_FLAG_FRAME_MBS_ONLY 0x10
#define V4L2_H264_SPS_FLAG_MB_ADAPTIVE_FRAME_FIELD 0x20
#define V4L2_H264_SPS_FLAG_DIRECT_8X8_INFERENCE 0x40
struct v4l2_ctrl_h264_sps {
  __u8 profile_idc;
  __u8 constraint_set_flags;
  __u8 level_idc;
  __u8 seq_parameter_set_id;
  __u8 chroma_format_idc;
  __u8 bit_depth_luma_minus8;
  __u8 bit_depth_chroma_minus8;
  __u8 log2_max_frame_num_minus4;
  __u8 pic_order_cnt_type;
  __u8 log2_max_pic_order_cnt_lsb_minus4;
  __s32 offset_for_non_ref_pic;
  __s32 offset_for_top_to_bottom_field;
  __u8 num_ref_frames_in_pic_order_cnt_cycle;
  __s32 offset_for_ref_frame[255];
  __u8 max_num_ref_frames;
  __u16 pic_width_in_mbs_minus1;
  __u16 pic_height_in_map_units_minus1;
  __u8 flags;
};

#define V4L2_H264_PPS_FLAG_ENTROPY_CODING_MODE 0x0001
#define V4L2_H264_PPS_FLAG_BOTTOM_FIELD_PIC_ORDER_IN_FRAME_PRESENT 0x0002
#define V4L2_H264_PPS_FLAG_WEIGHTED_PRED 0x0004
#define V4L2_H264_PPS_FLAG_DEBLOCKING_FILTER_CONTROL_PRESENT 0x0008
#define V4L2_H264_PPS_FLAG_CONSTRAINED_INTRA_PRED 0x0010
#define V4L2_H264_PPS_FLAG_REDUNDANT_PIC_CNT_PRESENT 0x0020
#define V4L2_H264_PPS_FLAG_TRANSFORM_8X8_MODE 0x0040
#define V4L2_H264_PPS_FLAG_PIC_SCALING_MATRIX_PRESENT 0x0080
struct v4l2_ctrl_h264_pps {
  __u8 pic_parameter_set_id;
  __u8 seq_parameter_set_id;
  __u8 num_slice_groups_minus1;
  __u8 num_ref_idx_l0_default_active_minus1;
  __u8 num_ref_idx_l1_default_active_minus1;
  __u8 weighted_bipred_idc;
  __s8 pic_init_qp_minus26;
  __s8 pic_init_qs_minus26;
  __s8 chroma_qp_index_offset;
  __s8 second_chroma_qp_index_offset;
  __u8 flags;
};

struct v4l2_ctrl_h264_scaling_matrix {
  __u8 scaling_list_4x4[6][16];
  __u8 scaling_list_8x8[6][64];
};

struct v4l2_h264_weight_factors {
  __s8 luma_weight[32];
  __s8 luma_offset[32];
  __s8 chroma_weight[32][2];
  __s8 chroma_offset[32][2];
};

struct v4l2_h264_pred_weight_table {
  __u8 luma_log2_weight_denom;
  __u8 chroma_log2_weight_denom;
  struct v4l2_h264_weight_factors weight_factors[2];
};

#define V4L2_SLICE_FLAG_FIELD_PIC 0x01
#define V4L2_SLICE_FLAG_BOTTOM_FIELD 0x02
#define V4L2_SLICE_FLAG_DIRECT_SPATIAL_MV_PRED 0x04
#define V4L2_SLICE_FLAG_SP_FOR_SWITCH 0x08
struct v4l2_ctrl_h264_slice_param {
  /* Size in bytes, including header */
  __u32 size;
  /* Offset in bits to slice_data() from the beginning of this slice. */
  __u32 header_bit_size;

  __u16 first_mb_in_slice;
  __u8 slice_type;
  __u8 pic_parameter_set_id;
  __u8 colour_plane_id;
  __u16 frame_num;
  __u16 idr_pic_id;
  __u16 pic_order_cnt_lsb;
  __s32 delta_pic_order_cnt_bottom;
  __s32 delta_pic_order_cnt0;
  __s32 delta_pic_order_cnt1;
  __u8 redundant_pic_cnt;

  struct v4l2_h264_pred_weight_table pred_weight_table;
  /* Size in bits of dec_ref_pic_marking() syntax element. */
  __u32 dec_ref_pic_marking_bit_size;
  /* Size in bits of pic order count syntax. */
  __u32 pic_order_cnt_bit_size;

  __u8 cabac_init_idc;
  __s8 slice_qp_delta;
  __s8 slice_qs_delta;
  __u8 disable_deblocking_filter_idc;
  __s8 slice_alpha_c0_offset_div2;
  __s8 slice_beta_offset_div2;
  __u32 slice_group_change_cycle;

  __u8 num_ref_idx_l0_active_minus1;
  __u8 num_ref_idx_l1_active_minus1;
  /*  Entries on each list are indices
   *  into v4l2_ctrl_h264_decode_param.dpb[]. */
  __u8 ref_pic_list0[32];
  __u8 ref_pic_list1[32];

  __u8 flags;
};

/* If not set, this entry is unused for reference. */
#define V4L2_H264_DPB_ENTRY_FLAG_ACTIVE 0x01
#define V4L2_H264_DPB_ENTRY_FLAG_LONG_TERM 0x02
struct v4l2_h264_dpb_entry {
  __u32 buf_index; /* v4l2_buffer index */
  __u16 frame_num;
  __u16 pic_num;
  /* Note that field is indicated by v4l2_buffer.field */
  __s32 top_field_order_cnt;
  __s32 bottom_field_order_cnt;
  __u8 flags; /* V4L2_H264_DPB_ENTRY_FLAG_* */
};

struct v4l2_ctrl_h264_decode_param {
  __u32 num_slices;
  __u8 idr_pic_flag;
  __u8 nal_ref_idc;
  __s32 top_field_order_cnt;
  __s32 bottom_field_order_cnt;
  __u8 ref_pic_list_p0[32];
  __u8 ref_pic_list_b0[32];
  __u8 ref_pic_list_b1[32];
  struct v4l2_h264_dpb_entry dpb[16];
};

#define V4L2_VP8_SEGMNT_HDR_FLAG_ENABLED 0x01
#define V4L2_VP8_SEGMNT_HDR_FLAG_UPDATE_MAP 0x02
#define V4L2_VP8_SEGMNT_HDR_FLAG_UPDATE_FEATURE_DATA 0x04
struct v4l2_vp8_sgmnt_hdr {
  __u8 segment_feature_mode;

  __s8 quant_update[4];
  __s8 lf_update[4];
  __u8 segment_probs[3];

  __u8 flags;
};

#define V4L2_VP8_LF_HDR_ADJ_ENABLE 0x01
#define V4L2_VP8_LF_HDR_DELTA_UPDATE 0x02
struct v4l2_vp8_loopfilter_hdr {
  __u8 type;
  __u8 level;
  __u8 sharpness_level;
  __s8 ref_frm_delta_magnitude[4];
  __s8 mb_mode_delta_magnitude[4];

  __u8 flags;
};

struct v4l2_vp8_quantization_hdr {
  __u8 y_ac_qi;
  __s8 y_dc_delta;
  __s8 y2_dc_delta;
  __s8 y2_ac_delta;
  __s8 uv_dc_delta;
  __s8 uv_ac_delta;
  __u16 dequant_factors[4][3][2];
};

struct v4l2_vp8_entropy_hdr {
  __u8 coeff_probs[4][8][3][11];
  __u8 y_mode_probs[4];
  __u8 uv_mode_probs[3];
  __u8 mv_probs[2][19];
};

#define V4L2_VP8_FRAME_HDR_FLAG_EXPERIMENTAL 0x01
#define V4L2_VP8_FRAME_HDR_FLAG_SHOW_FRAME 0x02
#define V4L2_VP8_FRAME_HDR_FLAG_MB_NO_SKIP_COEFF 0x04
struct v4l2_ctrl_vp8_frame_hdr {
  /* 0: keyframe, 1: not a keyframe */
  __u8 key_frame;
  __u8 version;

  /* Populated also if not a key frame */
  __u16 width;
  __u8 horizontal_scale;
  __u16 height;
  __u8 vertical_scale;

  struct v4l2_vp8_sgmnt_hdr sgmnt_hdr;
  struct v4l2_vp8_loopfilter_hdr lf_hdr;
  struct v4l2_vp8_quantization_hdr quant_hdr;
  struct v4l2_vp8_entropy_hdr entropy_hdr;

  __u8 sign_bias_golden;
  __u8 sign_bias_alternate;

  __u8 prob_skip_false;
  __u8 prob_intra;
  __u8 prob_last;
  __u8 prob_gf;

  __u32 first_part_size;
  __u32 first_part_offset;
  /*
   * Offset in bits of MB data in first partition,
   * i.e. bit offset starting from first_part_offset.
   */
  __u32 macroblock_bit_offset;

  __u8 num_dct_parts;
  __u32 dct_part_sizes[8];

  __u8 bool_dec_range;
  __u8 bool_dec_value;
  __u8 bool_dec_count;

  /* v4l2_buffer indices of reference frames */
  __u32 last_frame;
  __u32 golden_frame;
  __u32 alt_frame;

  __u8 flags;
};

/** @endcond */

/*---------------Add below NVIDIA specific extensions ----------------------*/

/**
 * @defgroup V4L2Dec V4L2 Video Decoder
 *
 * @brief NVIDIA Tegra V4L2 Video Decoder Description and Extensions
 *
 * The video decoder device node is \b "/dev/nvhost-nvdec".
 *
 * ### Supported pixelformats:
 * OUTPUT PLANE       | CAPTURE PLANE
 * :----------------: | :----------------:
 * V4L2_PIX_FMT_H264  | V4L2_PIX_FMT_NV12M
 * V4L2_PIX_FMT_H265  | V4L2_PIX_FMT_NV12M
 *
 * ### Supported memory types:
 * MEMORY               | OUTPUT PLANE | CAPTURE PLANE
 * :------------------: | :----------: | :-----------:
 * V4L2_MEMORY_MMAP     | Y            | Y
 * V4L2_MEMORY_DMABUF   | N            | N
 * V4L2_MEMORY_USERPTR  | N            | N
 *
 * ### Supported controls:
 * - #V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT
 * - #V4L2_CID_MPEG_VIDEO_DISABLE_DPB
 * - #V4L2_CID_MPEG_VIDEO_ERROR_REPORTING
 * - #V4L2_CID_MPEG_VIDEO_SKIP_FRAMES
 * - V4L2_CID_MIN_BUFFERS_FOR_CAPTURE (Get the minimum buffers to be allocated on capture plane.
 * Read only. Valid after #V4L2_EVENT_RESOLUTION_CHANGE)
 * - #V4L2_CID_MPEG_VIDEODEC_METADATA
 *
 * ### Supported events:
 * Event                         | Purpose
 * ----------------------------- | :----------------------------:
 * #V4L2_EVENT_RESOLUTION_CHANGE | Resolution of the stream has changed.
 *
 *
 * ### Handling resolution change events:
 * When the decoder generates a V4L2_EVENT_RESOLUTION_CHANGE event, the
 * application should call STREAMOFF on the capture plane, tell decoder to
 * deallocate the current buffers by calling REQBUF with \a count zero, get
 * the new capture plane format, and then proceed with setting up the buffers
 * for the capture plane.
 *
 * In case of decoder, the buffer format might differ from the Display resolution.
 * The application should use \b VIDIOC_G_CROP to get the display resolution.
 *
 * ### EOS Handling:
 * Following sequence should be followed for sending EOS and recieving EOS
 * from the decoder.
 * -# Send EOS to decoder by queueing on the output plane a buffer with
 * bytesused = 0 for the 0th plane (`v4l2_buffer.m.planes[0].bytesused = 0`).
 * -# Dequeue buffers on the output plane, till it gets a buffer with bytesused = 0
 * for the 0th plane (`v4l2_buffer.m.planes[0].bytesused == 0`)
 * -# Dequeue buffers on the capture plane, till it gets a buffer with bytesused = 0
 * for the 0th plane.
 *
 * ### Decoder output frame metadata:
 * Decoder supports reporting frame related metadata including error reports and
 * DPB info. See #V4L2_CID_MPEG_VIDEO_ERROR_REPORTING, #V4L2_CID_MPEG_VIDEODEC_METADATA
 * and #v4l2_ctrl_video_metadata for more information.
 */
/** @{ */

/**
 * Control ID to indicate to the decoder that the input buffers will not contain
 * complete buffers.
 *
 * @note This control should be set in case of frames containing multiple slices
 * when the input buffers do not contain all the slices of the frame.
 *
 * A boolean value should be supplied with this control.
 *
 * @attention This control should be set before setting formats on either plane.
 */
#define V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT (V4L2_CID_MPEG_BASE + 515)

/**
 * Control ID to disable decoder DPB management.
 *
 * @note This will only work for streams having a single reference frame.
 *
 * A boolean value should be supplied with this control.
 *
 * @attention This control should be set after setting formats on both the planes
 * and before requesting buffers on either plane.
 */
#define V4L2_CID_MPEG_VIDEO_DISABLE_DPB (V4L2_CID_MPEG_BASE + 516)

/**
 * Control ID to enable decoder error and metadata reporting.
 *
 * A boolean value should be supplied with this control.
 *
 * @attention This control should be set after setting formats on both the planes
 * and before requesting buffers on either plane.
 */
#define V4L2_CID_MPEG_VIDEO_ERROR_REPORTING (V4L2_CID_MPEG_BASE + 517)

/**
 * Control ID to set the skip frames property of the decoder.
 *
 * Decoder can be configured to skip certain types of frames. One of
 * #v4l2_skip_frames_type should be passed.
 *
 * @attention This control should be set after setting formats on both the planes
 * and before requesting buffers on either plane.
 * This control ID is supported only for H264.
 */
#define V4L2_CID_MPEG_VIDEO_SKIP_FRAMES (V4L2_CID_MPEG_BASE + 518)

/**
 * Control ID to get the decoder output metadata.
 *
 * @note Metadata reporting should be enabled using
 * #V4L2_CID_MPEG_VIDEO_ERROR_REPORTING ioctl for this.
 *
 * A pointer to a valid #v4l2_ctrl_video_metadata structure should be supplied
 * with this control.
 *
 * @attention This control should be read after dequeueing a buffer successfully from
 * the capture plane. The values in the structure are valid till the buffer is queued
 * again.
 */
#define V4L2_CID_MPEG_VIDEODEC_METADATA (V4L2_CID_MPEG_BASE + 519)

/** @} */

/**
 * @defgroup V4L2Conv V4L2 Video Converter
 *
 * @brief NVIDIA Tegra V4L2 Video Converter Description and Extensions
 *
 * Video converter can be used for color space conversion, scaling and
 * conversion between hardware buffer memory (V4L2_MEMORY_MMAP/
 * V4L2_MEMORY_DMABUF) and software buffer memory (V4L2_MEMORY_USERPTR) and
 * other operations such as cropping, flipping/rotating and
 * temporal noise reduction (TNR).
 * The video converter device node is \b "/dev/nvhost-vic".
 *
 * ### Supported pixelformats:
 *  PIXEL FORMAT           | PIXEL FORMAT
 * :---------------------: | :--------------:
 * V4L2_PIX_FMT_YUV444M    | V4L2_PIX_FMT_YVU422M
 * V4L2_PIX_FMT_YUV420M    | V4L2_PIX_FMT_YVU420M
 * V4L2_PIX_FMT_NV12M      | V4L2_PIX_FMT_GREY
 * V4L2_PIX_FMT_YUYV       | V4L2_PIX_FMT_YVYU
 * V4L2_PIX_FMT_UYVY       | V4L2_PIX_FMT_VYUY
 * V4L2_PIX_FMT_ABGR32     | V4L2_PIX_FMT_XBGR32
 *
 * ### Supported pixelformats for TNR:
 *  PIXEL FORMAT           | PIXEL FORMAT
 * :---------------------: | :--------------:
 * V4L2_PIX_FMT_YUV420M    | V4L2_PIX_FMT_NV12M
 * V4L2_PIX_FMT_UYVY       | V4L2_PIX_FMT_YUYV
 *
 * ### Supported memory types:
 * MEMORY               | OUTPUT PLANE | CAPTURE PLANE
 * :------------------: | :----------: | :-----------:
 * V4L2_MEMORY_MMAP     | Y            | Y
 * V4L2_MEMORY_DMABUF   | Y            | Y
 * V4L2_MEMORY_USERPTR  | Y            | Y
 *
 * ### Supported controls:
 * - #V4L2_CID_VIDEO_CONVERT_OUTPUT_PLANE_LAYOUT
 * - #V4L2_CID_VIDEO_CONVERT_CAPTURE_PLANE_LAYOUT
 * - #V4L2_CID_VIDEO_CONVERT_FLIP_METHOD
 * - #V4L2_CID_VIDEO_CONVERT_INTERPOLATION_METHOD
 * - #V4L2_CID_VIDEO_CONVERT_TNR_ALGORITHM
 *
 * ### Cropping:
 * Video converter supports cropping using VIDIOC_S_SELECTION ioctl with type
 * @a V4L2_BUF_TYPE_VIDEO_CAPTURE and target @a V4L2_SEL_TGT_CROP. This should
 * be set before requesting buffers on either plane.
 *
 * ### EOS Handling:
 * Following sequence should be followed for sending EOS and recieving EOS
 * from the converter.
 * -# Send EOS to converter by queueing on the output plane a buffer with
 * bytesused = 0 for the 0th plane (`v4l2_buffer.m.planes[0].bytesused = 0`).
 * -# Dequeue buffers on the capture plane, till it gets a buffer with bytesused = 0
 * for the 0th plane.
 */
/** @{ */

/**
 * Control ID to set converter output plane buffer layout.
 *
 * A value of type #v4l2_nv_buffer_layout should be supplied with this control.
 *
 * @attention This control should be set before requesting buffers on the output plane.
 */
#define V4L2_CID_VIDEO_CONVERT_OUTPUT_PLANE_LAYOUT (V4L2_CID_MPEG_BASE + 520)

/**
 * Control ID to set converter capture plane buffer layout.
 *
 * A value of type #v4l2_nv_buffer_layout should be supplied with this control.
 *
 * @attention This control should be set before requesting buffers on the capture plane.
 */
#define V4L2_CID_VIDEO_CONVERT_CAPTURE_PLANE_LAYOUT (V4L2_CID_MPEG_BASE + 521)

/**
 * Control ID to set the converter flip/rotation method.
 *
 * A value of type #v4l2_flip_method should be supplied with this control.
 *
 * @attention This control should be set before requesting buffers on either plane.
 */
#define V4L2_CID_VIDEO_CONVERT_FLIP_METHOD (V4L2_CID_MPEG_BASE + 522)

/**
 * Control ID to set the converter interpolation method.
 *
 * A value of type #v4l2_interpolation_method should be supplied with this control.
 *
 * @attention This control should be set before requesting buffers on either plane.
 */
#define V4L2_CID_VIDEO_CONVERT_INTERPOLATION_METHOD (V4L2_CID_MPEG_BASE + 523)

/**
 * Control ID to set the converter Temporal Noise Reduction (TNR) Algorithm.
 *
 * A value of type #v4l2_tnr_algorithm should be supplied with this control.
 *
 * @attention This control should be set before requesting buffers on either plane.
 * @attention TNR algorithms are not supported with YUV422 and YUV444 capture
 *            plane formats.
 */
#define V4L2_CID_VIDEO_CONVERT_TNR_ALGORITHM (V4L2_CID_MPEG_BASE + 524)
/** @} */

/**
 * @defgroup V4L2Enc V4L2 Video Encoder
 *
 * @brief NVIDIA Tegra V4L2 Video Encoder Description and Extensions
 *
 * The video encoder device node is \b "/dev/nvhost-msenc".
 *
 * ### Supported pixelformats:

 * OUTPUT PLANE            | CAPTURE PLANE
 * :---------------------: | :--------------
 * V4L2_PIX_FMT_YUV420M    | V4L2_PIX_FMT_H264
 *           -             | V4L2_PIX_FMT_H265
 *
 * ### Supported memory types:

 * MEMORY               | OUTPUT PLANE | CAPTURE PLANE
 * :------------------: | :----------: | :-----------:
 * V4L2_MEMORY_MMAP     | Y            | Y
 * V4L2_MEMORY_DMABUF   | Y            | N
 * V4L2_MEMORY_USERPTR  | N            | N
 * \attention For the video encoder, it is necessary that the capture plane
 *  format be set before the output plane format and only then request buffers on
 *  any of the planes.
 *
 * ### Supported controls:

 * #### Controls from the open souce v4l2-controls header:
 * Control Id                       | Purpose              | Runtime Configurable
 * -------------------------------- | -------------------- | :------------------:
 * V4L2_CID_MPEG_VIDEO_BITRATE      | Bitrate              | Y
 * V4L2_CID_MPEG_VIDEO_H264_PROFILE | H.264 Encode Profile | N
 * V4L2_CID_MPEG_VIDEO_BITRATE_MODE | Rate Control Mode    | N
 * V4L2_CID_MPEG_VIDEO_GOP_SIZE     | I-frame Interval     | N
 * V4L2_CID_MPEG_VIDEO_H264_LEVEL   | Encode Level         | N
 * V4L2_CID_MPEG_MFC51_VIDEO_FORCE_FRAME_TYPE | Force I-frame on one of queued output plane buffer | Y
 *
 * All non-runtime configurable options should be set after setting formats on
 * both the planes and before requesting buffers on either plane.
 *
 * The runtime configurable parameters can be called anytime after setting
 * formats on both the planes.
 *
 * #### NVidia specific controls:
 * - #V4L2_CID_MPEG_VIDEO_H265_PROFILE
 * - #V4L2_CID_MPEG_VIDEO_IDR_INTERVAL
 * - #V4L2_CID_MPEG_VIDEOENC_TEMPORAL_TRADEOFF_LEVEL
 * - #V4L2_CID_MPEG_VIDEOENC_SLICE_LENGTH_PARAM
 * - #V4L2_CID_MPEG_VIDEOENC_ROI_PARAMS
 * - #V4L2_CID_MPEG_VIDEOENC_VIRTUALBUFFER_SIZE
 * - #V4L2_CID_MPEG_VIDEOENC_NUM_REFERENCE_FRAMES
 * - #V4L2_CID_MPEG_VIDEOENC_SLICE_INTRAREFRESH_PARAM
 * - #V4L2_CID_MPEG_VIDEOENC_NUM_BFRAMES
 * - #V4L2_CID_MPEG_VIDEOENC_INSERT_SPS_PPS_AT_IDR
 * - #V4L2_CID_MPEG_VIDEOENC_METADATA
 * - #V4L2_CID_MPEG_VIDEOENC_METADATA_MV
 * - #V4L2_CID_MPEG_VIDEOENC_ENABLE_METADATA_MV
 * - #V4L2_CID_MPEG_VIDEOENC_QP_RANGE
 * - #V4L2_CID_MPEG_VIDEOENC_HW_PRESET_TYPE_PARAM
 * - #V4L2_CID_MPEG_VIDEOENC_INPUT_METADATA
 * - #V4L2_CID_MPEG_VIDEOENC_ENABLE_EXTERNAL_RPS_CONTROL
 * - #V4L2_CID_MPEG_VIDEOENC_ENABLE_EXTERNAL_RATE_CONTROL
 * - #V4L2_CID_MPEG_VIDEOENC_ENABLE_ROI_PARAM
 * - #V4L2_CID_MPEG_VIDEOENC_ENABLE_RECONCRC_PARAM
 *
 * #### Setting Framerate:
 * Encoder framerate can be set with VIDIOC_S_PARM ioctl by setting the numerator
 * and denominator in `v4l2_streamparm.parm.output.timeperframe`.
 *
 * ### Supported Encoder Profiles:
 * #### H.264
 * - V4L2_MPEG_VIDEO_H264_PROFILE_MAIN
 * - V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE
 * - V4L2_MPEG_VIDEO_H264_PROFILE_HIGH
 *
 * #### H.265
 * - V4L2_MPEG_VIDEO_H265_PROFILE_MAIN
 *
 * ### Encoder output metadata:
 * Encoder supports reporting frame related metadata including motion vectors
 * for that frame. See #V4L2_CID_MPEG_VIDEOENC_METADATA,
 * #V4L2_CID_MPEG_VIDEOENC_METADATA_MV and #V4L2_CID_MPEG_VIDEOENC_ENABLE_METADATA_MV
 * for more information.
 *
 * ### EOS Handling:
 * Following sequence should be followed for sending EOS and recieving EOS
 * from the encoder.
 * -# Send EOS to encoder by queueing on the output plane a buffer with
 * bytesused = 0 for the 0th plane (`v4l2_buffer.m.planes[0].bytesused = 0`).
 * -# Dequeue buffers on the capture plane, till it gets a buffer with bytesused = 0
 * for the 0th plane.
 */
/** @{ */

/**
 * Control ID to configure encoder to drop frames while encoding.
 *
 * A value of type #v4l2_enc_temporal_tradeoff_level_type should be supplied
 * with this control.
 *
 * @attention This control should be set after setting formats on both the planes
 * and before requesting buffers on either plane.
 */
#define V4L2_CID_MPEG_VIDEOENC_TEMPORAL_TRADEOFF_LEVEL (V4L2_CID_MPEG_BASE + 525)

/**
 * Control ID to configure encoder slice length either in terms of MBs or bits.
 *
 * A pointer to a valid #v4l2_enc_slice_length_param structure should be supplied
 * with this control.
 *
 * @attention This control should be set after setting formats on both the planes
 * and before requesting buffers on either plane.
 */
#define V4L2_CID_MPEG_VIDEOENC_SLICE_LENGTH_PARAM (V4L2_CID_MPEG_BASE + 526)

/**
 * Control ID to configure encoder to encode particular region of frame in high
 * quality.
 *
 * A pointer to a valid #v4l2_enc_frame_ROI_params structure should be supplied
 * with this control. This control will soon be deprecated instead use the
 * CID V4L2_CID_MPEG_VIDEOENC_INPUT_METADATA for sending the ROI parameters
 *
 * @attention This control should be set after requesting buffers on both the
 * planes.
 */
#define V4L2_CID_MPEG_VIDEOENC_ROI_PARAMS (V4L2_CID_MPEG_BASE + 527)

/**
 * Control ID to specify virtual buffer size in bits for encoder.
 *
 * A pointer to a valid #v4l2_enc_virtual_buffer_size structure should be
 * supplied with this control.
 *
 * @attention This control should be set after setting formats on both the planes
 * and before requesting buffers on either plane.
 */
#define V4L2_CID_MPEG_VIDEOENC_VIRTUALBUFFER_SIZE (V4L2_CID_MPEG_BASE + 528)

/**
 * Control ID to specify maximum number of reference frames that can be used.
 *
 * An integer value should be supplied with this control.
 *
 * @attention This control should be set after setting formats on both the planes
 * and before requesting buffers on either plane.
 */
#define V4L2_CID_MPEG_VIDEOENC_NUM_REFERENCE_FRAMES (V4L2_CID_MPEG_BASE + 529)

/**
 * Control ID to specify the encoder slice intra refresh interval.
 *
 * A pointer to a valid #v4l2_enc_slice_intrarefresh_param structure should be
 * supplied with this control.
 *
 * @attention This control should be set after setting formats on both the planes
 * and before requesting buffers on either plane.
 */
#define V4L2_CID_MPEG_VIDEOENC_SLICE_INTRAREFRESH_PARAM (V4L2_CID_MPEG_BASE + 530)

/**
 * Control ID to set number of B frames to be encoded between two P frames.
 *
 * Works only with H.264 encoder. An integer value should be supplied with this
 * control.
 *
 * @attention This control should be set after setting formats on both the planes
 * and before requesting buffers on either plane.
 */
#define V4L2_CID_MPEG_VIDEOENC_NUM_BFRAMES (V4L2_CID_MPEG_BASE + 531)

/**
 * Control ID to enable/disable inserting SPS and PPS explicitly at IDR interval.
 *
 * A boolean value should be supplied with this control.
 *
 * @attention This control should be set after setting formats on both the planes
 * and before requesting buffers on either plane.
 */
#define V4L2_CID_MPEG_VIDEOENC_INSERT_SPS_PPS_AT_IDR (V4L2_CID_MPEG_BASE + 532)

/**
 * Control ID to get encoder output metadata.
 *
 * A pointer to valid #v4l2_ctrl_video_metadata structure should be supplied with
 * this control.
 *
 * @attention This control should be read after dequeueing a buffer successfully from
 * the capture plane. The values in the structure are valid till the buffer is queued
 * again.
 */
#define V4L2_CID_MPEG_VIDEOENC_METADATA (V4L2_CID_MPEG_BASE + 533)

/**
 * Control ID to enable/disable encoder Motion Vector reporting.
 *
 * A boolean value should be supplied with this control.
 *
 * @attention This control should be set after setting formats on both the planes
 * and before requesting buffers on either plane.
 */
#define V4L2_CID_MPEG_VIDEOENC_ENABLE_METADATA_MV (V4L2_CID_MPEG_BASE + 534)

/**
 * Control ID to get encoder output motion vector metadata.
 *
 * A pointer to valid #v4l2_ctrl_videoenc_outputbuf_metadata_MV structure should
 * be supplied with this control.
 *
 * @attention This control should be read after dequeueing a buffer successfully from
 * the capture plane. The values in the structure are valid till the buffer is queued
 * again.
 */
#define V4L2_CID_MPEG_VIDEOENC_METADATA_MV (V4L2_CID_MPEG_BASE + 535)

/**
 * Control ID to set QP range for I/P/B frames
 *
 * A pointer to valid #v4l2_ctrl_video_qp_range structure should
 * be supplied with this control.
 *
 * @attention This control should be set after setting formats on both the planes
 * and before requesting buffers on either plane.
 */
#define V4L2_CID_MPEG_VIDEOENC_QP_RANGE (V4L2_CID_MPEG_BASE + 536)

/**
 * Control ID to set encoder HW Preset type
 *
 * A pointer to valid #v4l2_enc_hw_preset_type_param structure should
 * be supplied with this control.
 *
 * @attention This control should be set after setting formats on both the planes
 * and before requesting buffers on either plane.
 */
#define V4L2_CID_MPEG_VIDEOENC_HW_PRESET_TYPE_PARAM (V4L2_CID_MPEG_BASE + 537)

/**
 * Control ID to provide input metadata for encoder buffer.
 *
 * A pointer to valid #v4l2_ctrl_videoenc_input_metadata structure should be
 * supplied with this control.
 *
 * @attention This control should be called before queueing a buffer on the output
 * plane. Use the bitwise OR of v4l2_enc_input_metadata_param in the
 * v4l2_ctrl_videoenc_input_metadata.metadata_flag to provide different input
 * metadata parameters in one s_ctrl call.
 */
#define V4L2_CID_MPEG_VIDEOENC_INPUT_METADATA (V4L2_CID_MPEG_BASE + 538)

/**
 * Control ID to configure encoder for external RPS control.
 *
 * A pointer to a valid #v4l2_enc_enable_ext_rps_ctr structure should be supplied
 * with this control.
 *
 * @attention This control should be set after requesting buffers on both the
 * planes.
 */
#define V4L2_CID_MPEG_VIDEOENC_ENABLE_EXTERNAL_RPS_CONTROL (V4L2_CID_MPEG_BASE + 539)

/**
 * Control ID to configure encoder for external rate control.
 *
 * A pointer to a valid #v4l2_enc_enable_ext_rate_ctr structure should be supplied
 * with this control.
 *
 * @attention This control should be set after requesting buffers on both the
 * planes.
 */
#define V4L2_CID_MPEG_VIDEOENC_ENABLE_EXTERNAL_RATE_CONTROL (V4L2_CID_MPEG_BASE + 540)

/**
 * Control ID to configure ROI encoding for a session
 *
 * A pointer to a valid #v4l2_enc_enable_roi_param structure should be supplied
 * with this control.
 *
 * @attention This control should be set after requesting buffers on both the
 * planes.
 */
#define V4L2_CID_MPEG_VIDEOENC_ENABLE_ROI_PARAM (V4L2_CID_MPEG_BASE + 541)

/**
 * Control ID to configure Reconstructed CRC for a session
 *
 * A pointer to a valid #v4l2_enc_enable_reconcrc_param structure should be supplied
 * with this control.
 *
 * @attention This control should be set after requesting buffers on both the
 * planes.
 */
#define V4L2_CID_MPEG_VIDEOENC_ENABLE_RECONCRC_PARAM (V4L2_CID_MPEG_BASE + 542)

/** @} */

/** @addtogroup V4L2Dec */
/** @{ */
/**
 * Enum v4l2_skip_frames_type, possible methods for decoder skip frames. */
enum v4l2_skip_frames_type {
  /** Do not skip any frame */
  V4L2_SKIP_FRAMES_TYPE_NONE = 0,
  /** Skip all non-reference frames */
  V4L2_SKIP_FRAMES_TYPE_NONREF = 1,
  /** Skip all frames except IDR */
  V4L2_SKIP_FRAMES_TYPE_DECODE_IDR_ONLY = 2,
};

/**
 * Holds the decoder error status metadata for the frame.
 */
typedef struct v4l2_ctrl_videodec_statusmetadata_ {
  /** Error types:
   *  bit 0: Fatal
   *  bit 1: MB level syntax
   *  bit 2: Missing Slice(s)
   *  bit 3: PrevFrameLostFlag */
  __u32 DecodeError;
  /** Number of Macro blocks decoded without error */
  __u32 DecodedMBs;
  /** Number of Macro blocks where error was concealed */
  __u32 ConcealedMBs;
  /** POC of the reference frame used for concealment */
  __u32 nConcealedFromPOC;
  /** Time required to decode the frame, in microseconds */
  __u32 FrameDecodeTime;
} v4l2_ctrl_videodec_statusmetadata;

/**
 * Holds the the frame specific metadata for a reference frame.
 */
typedef struct v4l2_ctrl_videodec_refframe_metadata_ {
  /** Boolean value indicating if the frame is present in DPB */
  __u32 bPresent;
  /** Boolean value indicating if the frame is an IDR */
  __u32 bIdrFrame;
  /** Boolean value indicating if the frame is a long term reference frame */
  __u32 bLTRefFrame;
  /** Boolean value indicating if it is a predicted frame */
  __u32 bPredicted;
  /** Picture order count of the frame */
  __u32 nPictureOrderCnt;
  /** Frame number. Resets to zero for an IDR frame */
  __u32 nFrameNum;
  /** Long Term Frame Index of the frame */
  __u32 nLTRFrameIdx;
} v4l2_ctrl_videodec_refframe_metadata;

/**
 * Holds the the frame specific metadata for the current frame.
 */
typedef struct v4l2_ctrl_videodec_currentframe_metadata_ {
  /** Boolean value indicating if the current frame is a reference frame */
  __u32 bRefFrame;
  /** Boolean value indicating if the current frame is an IDR */
  __u32 bIdrFrame;
  /** Boolean value indicating if the current frame is a long term reference frame */
  __u32 bLTRefFrame;
  /** Picture order count of the current frame */
  __u32 nPictureOrderCnt;
  /** Frame number. Resets to zero for an IDR frame */
  __u32 nFrameNum;
  /** Long Term Frame Index of the current frame */
  __u32 nLTRFrameIdx;
} v4l2_ctrl_videodec_currentframe_metadata;

/**
 * Holds the decoder DPB info metadata.
 */
typedef struct v4l2_ctrl_videodec_dpbinfometadata_ {
  /** Metadata for the current decoded frame */
  v4l2_ctrl_videodec_currentframe_metadata currentFrame;
  /** Number of active frames present in the DPB */
  __u32 nActiveRefFrames;
  /** An array of metadatas for the active frames in the DPB. Only @a
   *  nActiveRefFrames elements in the array are valid */
  v4l2_ctrl_videodec_refframe_metadata RPSList[16];
} v4l2_ctrl_videodec_dpbinfometadata;

/**
 * Holds H.264 specific decoder metadata for the frame.
 */
typedef struct v4l2_ctrl_h264dec_bufmetadata_ {
  /** Number of bits in the frame */
  __u32 nFrameNumBits;
  /** Type of frame:
   *  0 = B
   *  1 = P
   *  2 = I */
  __u32 FrameType;
  /** Current DPB info of the decoder */
  v4l2_ctrl_videodec_dpbinfometadata dpbInfo;
} v4l2_ctrl_h264dec_bufmetadata;

/**
 * Holds H.265 specific decoder metadata for the frame.
 */
typedef struct v4l2_ctrl_hevcdec_bufmetadata_ {
  /** Number of bits in the frame */
  __u32 nPocLsbBits;
  /** Type of frame:
   *  0 = B
   *  1 = P
   *  2 = I */
  __u32 FrameType;
  /** Current DPB info of the decoder */
  v4l2_ctrl_videodec_dpbinfometadata dpbInfo;
} v4l2_ctrl_hevcdec_bufmetadata;

/**
 * Holds the video decoder output metadata for a frame.
 */
typedef struct v4l2_ctrl_videodec_outputbuf_metadata_ {
  /** Color primaries */
  __u8 ucColorPrimaries;
  /** Transfer Characteristics */
  __u8 ucTransferCharacteristics;
  /** Matrix Coefficients */
  __u8 ucMatrixCoefficients;
  /** Boolean value indicating if @a FrameDecStats has valid contents */
  __u32 bValidFrameStatus;
  /** Frame decode statistics */
  v4l2_ctrl_videodec_statusmetadata FrameDecStats;
  /** Codec specific metadata for the frame */
  union {
    /** H.264 specific metadata */
    v4l2_ctrl_h264dec_bufmetadata H264DecParams;
    /** H.265 specific metadata */
    v4l2_ctrl_hevcdec_bufmetadata HEVCDecParams;
  } CodecParams;
} v4l2_ctrl_videodec_outputbuf_metadata;
/** @} */

/** @addtogroup V4L2Enc */
/** @{ */

/**
 * Enum specifying the types of encoder temporal tradeoff levels
 */
enum v4l2_enc_temporal_tradeoff_level_type {
  /** Do not drop any buffers */
  V4L2_ENC_TEMPORAL_TRADEOFF_LEVEL_DROPNONE = 0,
  /** Drop 1 in every 5 buffers */
  V4L2_ENC_TEMPORAL_TRADEOFF_LEVEL_DROP1IN5,
  /** Drop 1 in every 3 buffers */
  V4L2_ENC_TEMPORAL_TRADEOFF_LEVEL_DROP1IN3,
  /** Drop 1 in every 2 buffers */
  V4L2_ENC_TEMPORAL_TRADEOFF_LEVEL_DROP1IN2,
  /** Drop 2 in every 3 buffers */
  V4L2_ENC_TEMPORAL_TRADEOFF_LEVEL_DROP2IN3,
};

/**
 * Enum specifying the encoder HW Preset type.
 */
enum v4l2_enc_hw_preset_type {
  /** Encoder HWPreset with per frame encode time UltraFast */
  V4L2_ENC_HW_PRESET_ULTRAFAST = 1,
  /** Encoder HWPreset with per frame encode time Fast */
  V4L2_ENC_HW_PRESET_FAST,
  /** Encoder HWPreset with per frame encode time Medium */
  V4L2_ENC_HW_PRESET_MEDIUM,
  /** Encoder HWPreset with per frame encode time Slow */
  V4L2_ENC_HW_PRESET_SLOW,
};

/**
 * Holds encoder HW Preset type parameters.
 *
 * To be used with #V4L2_CID_MPEG_VIDEOENC_HW_PRESET_TYPE_PARAM ioctl.
 */
typedef struct v4l2_enc_hw_preset_type_param_ {
  /** Type in which the encoder hw preset is specified, one of type #v4l2_enc_hw_preset_type. */
  enum v4l2_enc_hw_preset_type hw_preset_type;
  /** Boolean value indicating if encoder set to max clock. */
  __u8 set_max_enc_clock;
} v4l2_enc_hw_preset_type_param;

/**
 * Enum specifying the type of slice length.
 */
enum v4l2_enc_slice_length_type {
  /** Slice size is specified in terms of number of bytes */
  V4L2_ENC_SLICE_LENGTH_TYPE_BITS = 0,
  /** Slice size is specified in terms of number of macroblocks */
  V4L2_ENC_SLICE_LENGTH_TYPE_MBLK,
};

/**
 * Enum specifying the input buffer metadata flag
 */
enum v4l2_enc_input_metadata_param {
  /** Input metadata structure contains ROI parameters  */
  V4L2_ENC_INPUT_ROI_PARAM_FLAG = 1,
  /** Input metadata structure contains GDR parameters  */
  V4L2_ENC_INPUT_GDR_PARAM_FLAG = 1 << 1,
  /** Input metadata structure contains External RPS parameters  */
  V4L2_ENC_INPUT_RPS_PARAM_FLAG = 1 << 2,
  /** Input metadata structure contains External RC parameters  */
  V4L2_ENC_INPUT_RC_PARAM_FLAG = 1 << 3,
  /** Input metadata structure contains ReconCRC parameters  */
  V4L2_ENC_INPUT_RECONCRC_PARAM_FLAG = 1 << 4,
};

/**
 * Holds encoder slice length parameters.
 *
 * To be used with #V4L2_CID_MPEG_VIDEOENC_SLICE_LENGTH_PARAM ioctl.
 */
typedef struct v4l2_enc_slice_length_param_ {
  /** Type in which the slice length is specified, one of type #v4l2_enc_slice_length_type. */
  enum v4l2_enc_slice_length_type slice_length_type;
  /** Size of the slice in either number of bytes or number of macro blocks */
  __u32 slice_length;
} v4l2_enc_slice_length_param;

/**
 * Holds encoder virtual buffer size parameters.
 *
 * To be used with #V4L2_CID_MPEG_VIDEOENC_VIRTUALBUFFER_SIZE ioctl.
 */
typedef struct v4l2_enc_virtual_buffer_size_ {
  /** Size of the virtual buffer, in bits */
  __u32 size;
} v4l2_enc_virtual_buffer_size;

/**
 * Holds encoder number of reference frame parameters.
 *
 * To be used with #V4L2_CID_MPEG_VIDEOENC_NUM_REFERENCE_FRAMES ioctl.
 *
 * This is not supported for H265
 */
typedef struct v4l2_enc_num_ref_frames_ {
  /** Number of reference frames */
  __u32 frames;
} v4l2_enc_num_ref_frames;

/**
 * Holds encoder slice intrareferesh parameters.
 *
 * To be used with #V4L2_CID_MPEG_VIDEOENC_SLICE_INTRAREFRESH_PARAM ioctl.
 */
typedef struct v4l2_enc_slice_intrarefresh_param_ {
  /** Slice intrarefresh interval, in number of slices */
  __u32 interval;
} v4l2_enc_slice_intrarefresh_param;

/**
 * Maximum number of ROI regions supported by the encoder.
 */
#define V4L2_MAX_ROI_REGIONS 8

/**
 * Holds the encoder quality parameters for a single ROI region.
 */
typedef struct v4l2_enc_ROI_param_ {
  /** Region of interest rectangle */
  struct v4l2_rect ROIRect;
  /** QP delta for the region */
  __s32 QPdelta;
} v4l2_enc_ROI_param;

/**
 * Holds the encoder frame ROI parameters.
 *
 * Should be used with #V4L2_CID_MPEG_VIDEOENC_ROI_PARAMS ioctl.
 */
typedef struct v4l2_enc_frame_ROI_params_ {
  /** Number of regions */
  __u32 num_ROI_regions;
  /** Array of indiviudal ROI parameters */
  v4l2_enc_ROI_param ROI_params[V4L2_MAX_ROI_REGIONS];
  /** Config store integer to which this control is to be applied.
   *  This should be same as the value of config store of v4l2_buffer to which
   *  the ROI params should be applied. */
  __u32 config_store;
} v4l2_enc_frame_ROI_params;

/**
 * Holds the motion vector parameters for a single macro block.
 */
typedef struct MVInfo_ {
  /** Number of pixels the macro block has moved in horizontal direction */
  __s32 mv_x : 16;
  /** Number of pixels the macro block has moved in vertical direction */
  __s32 mv_y : 14;
  /** Temporal hints used by hardware for Motion Estimation */
  __u32 weight : 2;
} MVInfo;

/**
 * Holds the motion vector parameters for one complete frame.
 */
typedef struct v4l2_ctrl_videoenc_outputbuf_metadata_MV_ {
  /** Size of the pMVInfo buffer, in bytes */
  __u32 bufSize;
  /** Pointer to the buffer containing the motion vectors */
  MVInfo *pMVInfo;
} v4l2_ctrl_videoenc_outputbuf_metadata_MV;

/**
 * Maximum number of reference frames supported by the encoder.
 */
#define V4L2_MAX_REF_FRAMES 8

/**
 * Holds the RPS List parameters of encoded frame.
 */
typedef struct v4l2_enc_frame_full_prop_ {
  /** Unique frame Id */
  __u32 nFrameId;
  /** Boolean value indicating if current frame is an IDR */
  __u8 bIdrFrame;
  /** Boolean value indicating if set Long Term Ref Flag */
  __u8 bLTRefFrame;
  /** Picture Order Count */
  __u32 nPictureOrderCnt;
  /** FrameNum */
  __u32 nFrameNum;
  /** LongTermFrameIdx of a picture */
  __u32 nLTRFrameIdx;
} v4l2_enc_frame_full_prop;

/**
 * Holds the encoder output metadata for a frame.
 *
 * To be used with #V4L2_CID_MPEG_VIDEOENC_METADATA ioctl.
 */
typedef struct v4l2_ctrl_videoenc_outputbuf_metadata_ {
  /** Boolean value indicating if current frame is a key frame */
  __u8 KeyFrame;
  /** Boolean value indicating end of frame in case of multi slice encoding */
  __u8 EndofFrame;
  /** Average QP value of the frame */
  __u16 AvgQP;
  /** Boolean value indicating if current frame is a golden or alternate frame */
  __u8 bIsGoldenOrAlternateFrame;
  /** CRC for Reconstructed frame */
  __u8 bValidReconCRC;
  /** Recon Y-frame CRC */
  __u32 ReconFrame_Y_CRC;
  /** Recon U-frame CRC */
  __u32 ReconFrame_U_CRC;
  /** Recon V-frame CRC */
  __u32 ReconFrame_V_CRC;
  /** Number of bits needed to encode the frame */
  __u32 EncodedFrameBits;
  /** Minumum QP value in the frame */
  __u32 FrameMinQP;
  /** Maximum QP value in the frame */
  __u32 FrameMaxQP;
  /** RPS Feedback */
  __u32 bRPSFeedback_status;
  /**  Reference frame id used for Motion Estimation of current frame,
       ignored for IDR */
  __u32 nCurrentRefFrameId;
  /** Number of active reference frames */
  __u32 nActiveRefFrames;
  /** RPS List including most recent frame if it is reference frame */
  v4l2_enc_frame_full_prop RPSList[V4L2_MAX_REF_FRAMES];
} v4l2_ctrl_videoenc_outputbuf_metadata;

/**
 * Holds the metadata parameters for video encoder and decoder.
 *
 * The metadata is valid for the buffer with index @a buffer_index after the
 * buffer is dequeued till it is queued again.
 */
typedef struct v4l2_ctrl_video_metadata_ {
  /** A pointer to #v4l2_ctrl_videodec_outputbuf_metadata structure.
   * Should be a valid pointer when used with #V4L2_CID_MPEG_VIDEODEC_METADATA
   * ioctl. */
  v4l2_ctrl_videodec_outputbuf_metadata *VideoDecMetadata;
  /** A pointer to #v4l2_ctrl_videoenc_outputbuf_metadata structure.
   * Should be a valid pointer when used with #V4L2_CID_MPEG_VIDEOENC_METADATA
   * ioctl. */
  v4l2_ctrl_videoenc_outputbuf_metadata *VideoEncMetadata;
  /** A pointer to #v4l2_ctrl_videoenc_outputbuf_metadata_MV structure.
   * Should be a valid pointer when used with #V4L2_CID_MPEG_VIDEOENC_METADATA_MV
   * ioctl. */
  v4l2_ctrl_videoenc_outputbuf_metadata_MV *VideoEncMetadataMV;
  /** Index of the buffer whose metadata is required. */
  __u32 buffer_index;
} v4l2_ctrl_video_metadata;

/**
 * Holds the encoder GDR parameters.
 *
 * Should be used with #V4L2_CID_MPEG_VIDEOENC_INPUT_METADATA ioctl.
 */
typedef struct v4l2_enc_gdr_params_ {
  /** Parameter for GDR (Intra Refresh) for specified number of frames */
  __u32 nGDRFrames;
} v4l2_enc_gdr_params;

/**
 * Holds the params to configure encoder for external rps control.
 *
 * Should be used with #V4L2_CID_MPEG_VIDEOENC_ENABLE_EXTERNAL_RPS_CONTROL ioctl.
 */
typedef struct v4l2_enc_enable_ext_rps_ctrl_ {
  /** Boolean value indicating if enabled External RPS control */
  __u8 bEnableExternalRPS;
  /** Boolean value indicating if allowed gap in frame number */
  __u8 bGapsInFrameNumAllowed;
  /** TODO : Check for field details */
  __u32 nH264FrameNumBits;
  /** TODO : Check for field details */
  __u32 nH265PocLsbBits;
} v4l2_enc_enable_ext_rps_ctr;

/**
 * Holds the encoder frame property.
 */
typedef struct _v4l2_enc_frame_prop {
  /** unique Id */
  __u32 nFrameId;
  /** Long Term Ref Flag */
  __u8 bLTRefFrame;
} v4l2_enc_frame_prop;

/**
 * Holds the encoder frame external rps control parameters.
 *
 * Should be used with #V4L2_CID_MPEG_VIDEOENC_INPUT_METADATA ioctl.
 */
typedef struct v4l2_enc_frame_ext_rps_ctrl_params_ {
  /** unique Id of current frame */
  __u32 nFrameId;
  /** Boolean value indicating if current frame referenced or non-referenced */
  __u8 bRefFrame;
  /** Boolean value indicating if current frame long Term Ref Flag */
  __u8 bLTRefFrame;
  /** Max Number of reference frames to use for inter-motion search */
  __u32 nMaxRefFrames;
  /** # of valid entries in RPS, 0 means IDR */
  __u32 nActiveRefFrames;
  ;
  /**  frame id of reference frame to be used for motion search, ignored for IDR */
  __u32 nCurrentRefFrameId;
  /** Array of RPS */
  v4l2_enc_frame_prop RPSList[V4L2_MAX_REF_FRAMES];
} v4l2_enc_frame_ext_rps_ctrl_params;

/**
 * Holds the params to configure encoder for external rate control mode.
 *
 * Should be used with #V4L2_CID_MPEG_VIDEOENC_ENABLE_EXTERNAL_RATE_CONTROL ioctl.
 */
typedef struct v4l2_enc_enable_ext_rate_ctrl_ {
  /** Boolean value indicating if enabled External Picture RC */
  __u8 bEnableExternalPictureRC;
  /** Max QP per session when external picture RC enabled */
  __u32 nsessionMaxQP;
} v4l2_enc_enable_ext_rate_ctr;

/**
 * Holds the encoder frame external rate control parameters.
 *
 * Should be used with #V4L2_CID_MPEG_VIDEOENC_INPUT_METADATA ioctl.
 */
typedef struct v4l2_enc_frame_ext_rate_ctrl_params_ {
  /** Target frame bits */
  __u32 nTargetFrameBits;
  /** Frame start QP */
  __u32 nFrameQP;
  /** Frame min QP */
  __u32 nFrameMinQp;
  /** Frame max QP */
  __u32 nFrameMaxQp;
  /** Frame min QP deviation */
  __u32 nMaxQPDeviation;
} v4l2_enc_frame_ext_rate_ctrl_params;

/**
 * Holds the params to configure encoder for ROI parameters encoding
 *
 * Should be used with #V4L2_CID_MPEG_VIDEOENC_ENABLE_ROI_PARAM ioctl.
 */
typedef struct v4l2_enc_enable_roi_param_ {
  /** Boolean value to indicating ROI param encoding */
  __u8 bEnableROI;
} v4l2_enc_enable_roi_param;

/**
 * Holds the params to configure encoder for Reconstructed CRC encoding
 *
 * Should be used with #V4L2_CID_MPEG_VIDEOENC_ENABLE_RECONCRC_PARAM ioctl.
 */
typedef struct v4l2_enc_enable_reconcrc_param_ {
  /** Boolean value to indicating Reconstructed CRC encoding */
  __u8 bEnableReconCRC;
} v4l2_enc_enable_reconcrc_param;

/**
 * Holds the encoder frame Reconstructed CRC parameters.
 *
 * Should be used with #V4L2_CID_MPEG_VIDEOENC_INPUT_METADATA ioctl.
 */
typedef struct v4l2_enc_frame_ReconCRC_params_ {
  /** Rectangle to specify the co-ordinates of the input frame
   * used to calculate reconstructed picture CRC */
  struct v4l2_rect ReconCRCRect;
} v4l2_enc_frame_ReconCRC_params;

/**
 * Holds the encoder frame input metadata parameters.
 *
 * Should be used with #V4L2_CID_MPEG_VIDEOENC_INPUT_METADATA ioctl.
 */
typedef struct v4l2_ctrl_videoenc_input_metadata_ {
  /** Flag to indicate which inputbuffer metadata is valid. */
  __u32 flag;
  /** Pointer to the ROI params structure when ROI param is in metadata_flag. */
  v4l2_enc_frame_ROI_params *VideoEncROIParams;
  /** Pointer to the Reconstructed CRC parameter structure when ReconCRC param is in
   * metadata flag */
  v4l2_enc_frame_ReconCRC_params *VideoReconCRCParams;
  /** Pointer to the GDR params structure when GDR param is in metadata_flag. */
  v4l2_enc_gdr_params *VideoEncGDRParams;
  /** Pointer to the External RPL control parameter structure when RPS param is in
   * metadata flag */
  v4l2_enc_frame_ext_rps_ctrl_params *VideoEncRPSParams;
  /** Pointer to the External Rate control parameter structure when RC param is in
   * metadata flag */
  v4l2_enc_frame_ext_rate_ctrl_params *VideoEncExtRCParams;
  /** Config store integer to which these parameters are to be applied.
   *  This should be same as the value of config store of queued v4l2_buffer
   *   for which these parameters are valid. */
  __u32 config_store;
} v4l2_ctrl_videoenc_input_metadata;

/**
 * Setting Qp values in #v4l2_ctrl_video_qp_range to QP_RETAIN_VAL
 * will retain default or previously set QP values.
 */
#define QP_RETAIN_VAL -1

/**
 * Holds the encoder frame min/max QP parameters.
 *
 * Should be used with #V4L2_CID_MPEG_VIDEOENC_QP_RANGE ioctl.
 */
typedef struct _v4l2_ctrl_video_qp_range {
  /** Minimum QP value for I frame */
  __u32 MinQpI;
  /** Maximum QP value for I frame */
  __u32 MaxQpI;
  /** Minimum QP value for P frame */
  __u32 MinQpP;
  /** Maximum QP value for P frame */
  __u32 MaxQpP;
  /** Minimum QP value for B frame */
  __u32 MinQpB;
  /** Maximum QP value for B frame */
  __u32 MaxQpB;
} v4l2_ctrl_video_qp_range;
/** @} */

/** @addtogroup V4L2Conv */
/** @{ */

/**
 * Enum specifying types of buffer layouts.
 */
enum v4l2_nv_buffer_layout {
  V4L2_NV_BUFFER_LAYOUT_PITCH = 0,       /**< Pitch Linear Layout */
  V4L2_NV_BUFFER_LAYOUT_BLOCKLINEAR = 1, /**< Block Linear Layout */
};

/**
 * Enum specifying types of rotation/flip algorithms.
 */
enum v4l2_flip_method {
  V4L2_FLIP_METHOD_IDENTITY = 0, /**< Identity (no rotation) */
  V4L2_FLIP_METHOD_90L = 1,      /**< Rotate counter-clockwise 90 degrees */
  V4L2_FLIP_METHOD_180 = 2,      /**< Rotate 180 degrees */
  V4L2_FLIP_METHOD_90R = 3,      /**< Rotate clockwise 90 degrees */
  V4L2_FLIP_METHOD_HORIZ = 4,    /**< Flip horizontally */
  V4L2_FLIP_METHOD_INVTRANS = 5, /**< Flip across upper right/lower left diagonal */
  V4L2_FLIP_METHOD_VERT = 6,     /**< Flip vertically */
  V4L2_FLIP_METHOD_TRANS = 7,    /**< Flip across upper left/lower right diagonal */
};

/**
 * Enum specifying types of interpolation methods.
 */
enum v4l2_interpolation_method {
  V4L2_INTERPOLATION_NEAREST = 1,  /**< Nearest interpolation method */
  V4L2_INTERPOLATION_LINEAR = 2,   /**< Linear interpolation method */
  V4L2_INTERPOLATION_SMART = 3,    /**< Smart interpolation method */
  V4L2_INTERPOLATION_BILINEAR = 4, /**< Bi-Linear interpolation method */
};

/**
 * Enum specifying types of TNR algorithms.
 */
enum v4l2_tnr_algorithm {
  V4L2_TNR_ALGO_ORIGINAL = 0,             /**< Default TNR Algorithm */
  V4L2_TNR_ALGO_OUTDOOR_LOW_LIGHT = 1,    /**< Outdoor Low Light TNR Algorithm */
  V4L2_TNR_ALGO_OUTDOOR_MEDIUM_LIGHT = 2, /**< Outdoor Medium Light TNR Algorithm */
  V4L2_TNR_ALGO_OUTDOOR_HIGH_LIGHT = 3,   /**< Outdoor High Light TNR Algorithm */
  V4L2_TNR_ALGO_INDOOR_LOW_LIGHT = 4,     /**< Indoor Low Light TNR Algorithm */
  V4L2_TNR_ALGO_INDOOR_MEDIUM_LIGHT = 5,  /**< Indoor Medium Light TNR Algorithm */
  V4L2_TNR_ALGO_INDOOR_HIGH_LIGHT = 6,    /**< Indoor High Light TNR Algorithm */
};
/** @} */
#endif /*__V4L2_NV_EXTENSIONS_H__*/
