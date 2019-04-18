// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mergerMaskConfig.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/rigDef.hpp"
#include "libvideostitch/rigCameraDef.hpp"
#include "libvideostitch/cameraDef.hpp"

namespace VideoStitch {
namespace MergerMask {

MergerMaskConfig::MergerMaskConfig(const Ptv::Value* config) : isConfigValid(true) {
  if (!config) {
    isConfigValid = false;
    return;
  }

  // Parse list of video frames
  const Ptv::Value* val_list_frames = config->has("list_frames");
  frames.clear();
  frames.push_back(0);
  if (val_list_frames && val_list_frames->getType() == Ptv::Value::LIST) {
    std::vector<Ptv::Value*> listFramesPTV = val_list_frames->asList();
    frames.clear();
    for (auto& f : listFramesPTV) {
      frames.push_back((unsigned int)f->asInt());
    }
  }

  // Find mask max width
  const Ptv::Value* val_max_width = config->has("max_width");
  maxOverlappingWidth = -1000;
  if (val_max_width) {
    if (val_max_width->getType() == Ptv::Value::INT) {
      maxOverlappingWidth = (int)val_max_width->asInt();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Find size threshold
  const Ptv::Value* val_size_threshold = config->has("size_threshold");
  sizeThreshold = 1024;
  if (val_size_threshold) {
    if (val_size_threshold->getType() == Ptv::Value::INT) {
      sizeThreshold = (size_t)val_size_threshold->asInt();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Find kernel size
  const Ptv::Value* val_kernel_size = config->has("kernel_size");
  kernelSize = 1;
  if (val_kernel_size) {
    if (val_kernel_size->getType() == Ptv::Value::INT) {
      kernelSize = (int)val_kernel_size->asInt();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Find distortion threshold
  const Ptv::Value* val_distortion_threshold = config->has("distortion_threshold");
  distortionThreshold = 50;
  if (val_distortion_threshold) {
    if (val_distortion_threshold->getType() == Ptv::Value::INT) {
      distortionThreshold = (unsigned char)val_distortion_threshold->asInt();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Find distortion param
  const Ptv::Value* val_distortion_param = config->has("distortion_param");
  distortionParam = 3.0f;
  if (val_distortion_param) {
    if (val_distortion_param->getType() == Ptv::Value::DOUBLE) {
      distortionParam = (float)val_distortion_param->asDouble();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Find blending order param
  const Ptv::Value* val_blending_order_param = config->has("blending_order");
  blendingOrder = true;
  if (val_blending_order_param) {
    if (val_blending_order_param->getType() == Ptv::Value::BOOL) {
      blendingOrder = (bool)val_blending_order_param->asBool();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Find blending order param
  const Ptv::Value* val_seam_param = config->has("seam");
  seam = false;
  if (val_seam_param) {
    if (val_seam_param->getType() == Ptv::Value::BOOL) {
      seam = (bool)val_seam_param->asBool();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Find seam feathering size param
  const Ptv::Value* val_seam_feathering_size_param = config->has("seam_feathering_size");
  seamFeatheringSize = 20;
  if (val_seam_feathering_size_param) {
    if (val_seam_feathering_size_param->getType() == Ptv::Value::INT) {
      seamFeatheringSize = (int)val_seam_feathering_size_param->asInt();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Find seam feathering size param
  const Ptv::Value* val_input_scale_factor = config->has("input_scale_factor");
  inputScaleFactor = 2;
  if (val_input_scale_factor) {
    if (val_input_scale_factor->getType() == Ptv::Value::INT) {
      inputScaleFactor = (int)val_input_scale_factor->asInt();
    } else {
      isConfigValid = false;
      return;
    }
  }
}

MergerMaskConfig::MergerMaskConfig(const MergerMaskConfig& other)
    : isConfigValid(other.isConfigValid),
      maxOverlappingWidth(other.maxOverlappingWidth),
      blendingOrder(other.blendingOrder),
      seam(other.seam),
      sizeThreshold(other.sizeThreshold),
      distortionThreshold(other.distortionThreshold),
      distortionParam(other.distortionParam),
      kernelSize(other.kernelSize),
      seamFeatheringSize(other.seamFeatheringSize),
      inputScaleFactor(other.inputScaleFactor),
      frames(other.frames) {}

}  // namespace MergerMask
}  // namespace VideoStitch
