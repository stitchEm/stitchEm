// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/mergerMaskDef.hpp"

#include "parse/json.hpp"
#include "util/pngutil.hpp"
#include "util/strutils.hpp"
#include "util/compressionUtils.hpp"
#include "mask/mergerMask.hpp"
#include "panoInputDefsPimpl.hpp"

#include "libvideostitch/logging.hpp"

//#define MASK_COMPRESSION_DEBUG

#if defined(MASK_COMPRESSION_DEBUG)
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include "../util/pngutil.hpp"
#include "../util/pnm.hpp"
#include "../util/debugUtils.hpp"
#endif

namespace VideoStitch {
namespace Core {

MergerMaskDefinition::Pimpl::Pimpl()
    : maskOrders(0), width(0), height(0), inputScaleFactor(2), enabled(false), interpolationEnabled(false) {}

MergerMaskDefinition::Pimpl::~Pimpl() {}

void MergerMaskDefinition::InputIndexPixelData::clear() {
  // Clear old data
  width = 0;
  height = 0;
  datas.clear();
}

size_t MergerMaskDefinition::InputIndexPixelData::getFrameCount() const { return datas.size(); }

Status MergerMaskDefinition::InputIndexPixelData::realloc(const frameid_t frameId, const int64_t newWidth,
                                                          const int64_t newHeight,
                                                          const std::map<videoreaderid_t, std::string>& newData) {
  if (newWidth <= 0 || newHeight <= 0) {
    return {Origin::Input, ErrType::InvalidConfiguration, "Invalid panorama size (less than 1)."};
  }
  // If the size has been changed, release old memory
  if (newWidth != width || newHeight != height) {
    clear();
  }
  // Find the new size
  width = newWidth;
  height = newHeight;
  if (datas.find(frameId) == datas.end()) {
    datas.insert({frameId, newData});
  } else {
    datas[frameId] = newData;
  }
  return Status::OK();
}

MergerMaskDefinition::InputIndexPixelData::~InputIndexPixelData() {}

size_t MergerMaskDefinition::InputIndexPixelData::getDataCount(const frameid_t frameId) {
  if (datas.find(frameId) != datas.end()) {
    return datas[frameId].size();
  }
  return 0;
}

const std::string& MergerMaskDefinition::InputIndexPixelData::getData(const frameid_t frameId, const size_t index) {
  static std::string emptyString("");
  if (datas.find(frameId) == datas.end()) {
    return emptyString;
  }
  auto it = datas[frameId].find((int)index);
  if (it == datas[frameId].end()) {
    return emptyString;
  }
  return it->second;
}

const std::map<frameid_t, std::map<videoreaderid_t, std::string>>&
MergerMaskDefinition::InputIndexPixelData::getData() {
  return datas;
}

std::vector<videoreaderid_t> MergerMaskDefinition::InputIndexPixelData::getInputIndices() const {
  if (datas.size() == 0) {
    return std::vector<videoreaderid_t>();
  }
  auto data = datas.begin();
  std::vector<videoreaderid_t> indices;
  for (auto d : (*data).second) {
    indices.push_back(d.first);
  }
  return indices;
}

const std::map<videoreaderid_t, std::string>& MergerMaskDefinition::InputIndexPixelData::getData(
    const frameid_t frameId) {
  static std::map<videoreaderid_t, std::string> emptyStrings;
  if (datas.find(frameId) == datas.end()) {
    return emptyStrings;
  }
  return datas[frameId];
}

std::vector<frameid_t> MergerMaskDefinition::InputIndexPixelData::getFrameIds() const {
  std::vector<frameid_t> frameIds;
  for (auto imap : datas) {
    frameIds.push_back(imap.first);
  }
  return frameIds;
}

void MergerMaskDefinition::InputIndexPixelData::removeFrameIds(const std::vector<frameid_t>& frameIds) {
  for (auto frameId : frameIds) {
    datas.erase(frameId);
  }
}

std::pair<frameid_t, frameid_t> MergerMaskDefinition::InputIndexPixelData::getBoundedFrameIds(
    const frameid_t currentFrameId) const {
  frameid_t prevFrame, nextFrame;
  const std::vector<frameid_t> frameIds = getFrameIds();
  prevFrame = std::numeric_limits<frameid_t>::max();
  nextFrame = std::numeric_limits<frameid_t>::min();
  for (auto frameId : frameIds) {
    if (frameId <= currentFrameId) {
      if ((prevFrame == std::numeric_limits<frameid_t>::max()) || (frameId > prevFrame)) {
        prevFrame = frameId;
      }
    }
    if (frameId >= currentFrameId) {
      if ((nextFrame == std::numeric_limits<frameid_t>::min()) || (frameId < nextFrame)) {
        nextFrame = frameId;
      }
    }
  }
  return {prevFrame, nextFrame};
}

std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>>
MergerMaskDefinition::InputIndexPixelData::getFullData(const frameid_t currentFrameId) {
  std::pair<frameid_t, frameid_t> boundedFrames = getBoundedFrameIds(currentFrameId);
  const frameid_t prevFrame = boundedFrames.first;
  const frameid_t nextFrame = boundedFrames.second;
  if ((prevFrame == nextFrame) || (prevFrame < currentFrameId && nextFrame < currentFrameId)) {
    // This is the exact frame, just take the computed mask
    return std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>>{{prevFrame, getData(prevFrame)}};
  } else if (prevFrame > currentFrameId && nextFrame > currentFrameId) {
    // Use the first computed frame
    return std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>>{{nextFrame, getData(nextFrame)}};
  } else if (prevFrame < currentFrameId && nextFrame > currentFrameId) {
    // Current frame is in the middle of two computed frames. Interpolate the in-between result
    return std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>>{{prevFrame, getData(prevFrame)},
                                                                                     {nextFrame, getData(nextFrame)}};
  }
  return std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>>();
}

MergerMaskDefinition::MergerMaskDefinition() : pimpl(new Pimpl()) {}

MergerMaskDefinition::~MergerMaskDefinition() { delete pimpl; }

GENGETSETTER(MergerMaskDefinition, int64_t, Width, width)
GENGETSETTER(MergerMaskDefinition, int64_t, Height, height)
GENGETSETTER(MergerMaskDefinition, bool, Enabled, enabled)
GENGETSETTER(MergerMaskDefinition, bool, InterpolationEnabled, interpolationEnabled)
GENGETSETTER(MergerMaskDefinition, std::vector<size_t>, MasksOrder, maskOrders)
GENGETSETTER(MergerMaskDefinition, int, InputScaleFactor, inputScaleFactor)

MergerMaskDefinition* MergerMaskDefinition::create(const Ptv::Value& value) {
  std::unique_ptr<MergerMaskDefinition> res(new MergerMaskDefinition());
  const Ptv::Value* merger_mask_val = value.has("merger_mask");
  if (merger_mask_val) {
    if (!res->applyDiff(*merger_mask_val, true).ok()) {
      return nullptr;
    }
  }
  return res.release();
}

MergerMaskDefinition* MergerMaskDefinition::clone() const {
  MergerMaskDefinition* result = new MergerMaskDefinition();

#define AUTO_FIELD_COPY(field) result->set##field(get##field())
#define PIMPL_FIELD_COPY(field) result->pimpl->field = pimpl->field;

  PIMPL_FIELD_COPY(width);
  PIMPL_FIELD_COPY(height);
  PIMPL_FIELD_COPY(maskOrders);
  PIMPL_FIELD_COPY(enabled);
  PIMPL_FIELD_COPY(interpolationEnabled);
  PIMPL_FIELD_COPY(inputScaleFactor);
  std::vector<frameid_t> frameIds = pimpl->inputIndexPixelDataCache.getFrameIds();
  for (auto frameId : frameIds) {
    result->pimpl->inputIndexPixelDataCache.realloc(frameId, pimpl->inputIndexPixelDataCache.getWidth(),
                                                    pimpl->inputIndexPixelDataCache.getHeight(),
                                                    pimpl->inputIndexPixelDataCache.getData(frameId));
  }

  return result;
}

std::vector<frameid_t> MergerMaskDefinition::getFrameIds() const { return getInputIndexPixelData().getFrameIds(); }

void MergerMaskDefinition::removeFrameIds(const std::vector<frameid_t>& frameIds) {
  pimpl->inputIndexPixelDataCache.removeFrameIds(frameIds);
}

const MergerMaskDefinition::InputIndexPixelData& MergerMaskDefinition::getInputIndexPixelData() const {
  return pimpl->inputIndexPixelDataCache;
}

std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>>
MergerMaskDefinition::getInputIndexPixelDataIfValid(const frameid_t frameId) const {
  return validateInputIndexPixelData() ? pimpl->inputIndexPixelDataCache.getFullData(frameId)
                                       : std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>>();
}

bool MergerMaskDefinition::validateInputIndexPixelData() const {
  const InputIndexPixelData& mpd = getInputIndexPixelData();
  return mpd.getWidth() == getWidth() && mpd.getHeight() == getHeight() && mpd.getFrameCount() > 0;
}

Status MergerMaskDefinition::setInputIndexPixelData(const std::map<videoreaderid_t, std::string>& encodedMasks,
                                                    const uint64_t width, const uint64_t height,
                                                    const frameid_t frameId) {
#ifdef MASK_COMPRESSION_DEBUG
  {
    std::stringstream ss;
    ss.str("");
    ss << "original-mask.png";
    std::vector<uint32_t> v(buffer, buffer + width * height);
    Debug::dumpRGBAIndexDeviceBuffer<uint32_t>(ss.str().c_str(), v, width, height);
  }
#endif

#ifdef MASK_COMPRESSION_DEBUG
  {
    std::vector<uint32_t> v;
    Util::Compression::convertEncodedMasksToMask(width, height, encodedBuffer, v);
    std::stringstream ss;
    ss.str("");
    ss << "decoded-mask.png";
    Debug::dumpRGBAIndexDeviceBuffer<uint32_t>(ss.str().c_str(), v, width, height);
  }
#endif
  FAIL_RETURN(pimpl->inputIndexPixelDataCache.realloc(frameId, width, height, encodedMasks));
  return Status::OK();
}

Ptv::Value* MergerMaskDefinition::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("width", new Parse::JsonValue((int64_t)getWidth()));
  res->push("height", new Parse::JsonValue((int64_t)getHeight()));
  res->push("enable", new Parse::JsonValue(getEnabled()));
  res->push("interpolationEnabled", new Parse::JsonValue(getInterpolationEnabled()));
  res->push("inputScaleFactor", new Parse::JsonValue((int64_t)getInputScaleFactor()));
  auto mask_datas = std::make_unique<std::vector<Ptv::Value*>>();
  for (auto inputIndexData : pimpl->inputIndexPixelDataCache.getData()) {
    auto maskData = std::make_unique<std::vector<Ptv::Value*>>();
    for (size_t i = 0; i < pimpl->inputIndexPixelDataCache.getDataCount(inputIndexData.first); i++) {
      std::string encoded;
      encoded = inputIndexData.second[(int)i];
      maskData->push_back(new Parse::JsonValue(encoded));
    }
    if (maskData->size() > 0) {
      Ptv::Value* mask_data = Ptv::Value::emptyObject();
      mask_data->push(
          "input_index_count",
          new Parse::JsonValue((int64_t)pimpl->inputIndexPixelDataCache.getDataCount(inputIndexData.first)));
      mask_data->push("frameId", new Parse::JsonValue(inputIndexData.first));
      mask_data->push("input_index_data", new Parse::JsonValue(maskData.release()));
      mask_datas->push_back(mask_data);

      auto inputIndicesValues = std::make_unique<std::vector<Ptv::Value*>>();
      std::vector<videoreaderid_t> inputIndices = pimpl->inputIndexPixelDataCache.getInputIndices();
      for (size_t i = 0; i < inputIndices.size(); i++) {
        inputIndicesValues->push_back(new Parse::JsonValue((int64_t)inputIndices[i]));
      }
      if (inputIndicesValues->size() > 0) {
        mask_data->push("input_indices", new Parse::JsonValue(inputIndicesValues.release()));
      }
    }
  }
  res->push("masks", new Parse::JsonValue(mask_datas.release()));

  auto maskOrderValues = std::make_unique<std::vector<Ptv::Value*>>();
  std::vector<size_t> masksOrder = getMasksOrder();
  for (size_t i = 0; i < masksOrder.size(); i++) {
    maskOrderValues->push_back(new Parse::JsonValue((int64_t)masksOrder[i]));
  }
  if (maskOrderValues->size() > 0) {
    res->push("masks_order", new Parse::JsonValue(maskOrderValues.release()));
  }
  return res;
}

Status MergerMaskDefinition::applyDiff(const Ptv::Value& value, bool enforceMandatoryFields) {
  // Make sure value is an object.
  if (!Parse::checkType("merger_mask", value, Ptv::Value::OBJECT)) {
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
            "Invalid type for 'merger_mask' configuration, expected object"};
  }

#define PROPAGATE_NOK(config_name, varName)                                                                          \
  if (Parse::populateInt("MergerMaskDefinition", value, config_name, varName, enforceMandatoryFields) !=             \
      Parse::PopulateResult_Ok) {                                                                                    \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                                            \
            "Could not find valid '" config_name "' configuration in MergerMaskDefinition, expected integer value"}; \
  }
#define PROPAGATE_NOK_VAL(config_name, varName, val)                                                                 \
  if (Parse::populateInt("MergerMaskDefinition", val, config_name, varName, enforceMandatoryFields) !=               \
      Parse::PopulateResult_Ok) {                                                                                    \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                                            \
            "Could not find valid '" config_name "' configuration in MergerMaskDefinition, expected integer value"}; \
  }

  if (enforceMandatoryFields) {
    PROPAGATE_NOK("width", pimpl->width);
    PROPAGATE_NOK("height", pimpl->height);
    Parse::populateInt("MergerMaskDefinition", value, "inputScaleFactor", pimpl->inputScaleFactor, false);
    Parse::populateBool("MergerMaskDefinition", value, "interpolationEnabled", pimpl->interpolationEnabled, false);
    if (Parse::populateBool("MergerMaskDefinition", value, "enable", pimpl->enabled, enforceMandatoryFields) !=
        Parse::PopulateResult_Ok) {
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
              "Could not find valid 'enable' configuration in MergerMaskDefinition, expected boolean value"};
    }
    const Ptv::Value* val_masks = value.has("masks");
    if (val_masks && val_masks->getType() == Ptv::Value::LIST) {
      std::vector<Ptv::Value*> list_masks = val_masks->asList();
      for (auto& list_mask : list_masks) {
        int dataCount;
        int frameId;
        std::vector<int> inputIndices;
        const Ptv::Value* val_input_indices = list_mask->has("input_indices");
        if (val_input_indices && val_input_indices->getType() == Ptv::Value::LIST) {
          std::vector<Ptv::Value*> list_input_indices = val_input_indices->asList();
          for (auto& f : list_input_indices) {
            if (f->getType() == Ptv::Value::INT) {
              int inputIndex = (int)f->asInt();
              inputIndices.push_back(inputIndex);
            }
          }
        }

        PROPAGATE_NOK_VAL("input_index_count", dataCount, *list_mask);
        PROPAGATE_NOK_VAL("frameId", frameId, *list_mask);
        std::map<videoreaderid_t, std::string> encodeds;
        const Ptv::Value* val_input_index_data = list_mask->has("input_index_data");
        if (val_input_index_data && val_input_index_data->getType() == Ptv::Value::LIST) {
          std::vector<Ptv::Value*> list_input_index_data = val_input_index_data->asList();
          if (list_input_index_data.size() != inputIndices.size()) {
            return {Origin::BlendingMaskAlgorithm, ErrType::InvalidConfiguration, "Number of inputs does not matched"};
          }
          for (size_t i = 0; i < list_input_index_data.size(); i++) {
            const Ptv::Value* f = list_input_index_data[i];
            if (f->getType() == Ptv::Value::STRING) {
              std::string encoded = f->asString();
              encodeds[inputIndices[i]] = encoded;
            }
          }
        }
        FAIL_RETURN(pimpl->inputIndexPixelDataCache.realloc(frameId, pimpl->width, pimpl->height, encodeds));
      }
    }

    const Ptv::Value* val_masks_order = value.has("masks_order");
    std::vector<size_t> masksOrder;
    if (val_masks_order && val_masks_order->getType() == Ptv::Value::LIST) {
      std::vector<Ptv::Value*> list_masks_order = val_masks_order->asList();
      for (auto& f : list_masks_order) {
        if (f->getType() == Ptv::Value::INT) {
          masksOrder.push_back((size_t)f->asInt());
        }
      }
      setMasksOrder(masksOrder);
    }
#undef PROPAGATE_NOK
    return Status::OK();
  }
  return Status::OK();
}

}  // namespace Core
}  // namespace VideoStitch
