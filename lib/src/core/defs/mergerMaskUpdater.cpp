// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/mergerMaskUpdater.hpp"

namespace VideoStitch {
namespace Core {

MergerMaskDefinition* MergerMaskUpdater::clone() const { return mergerMaskDefinition->clone(); }

Ptv::Value* MergerMaskUpdater::serialize() const { return mergerMaskDefinition->serialize(); }

bool MergerMaskUpdater::getEnabled() const { return mergerMaskDefinition->getEnabled(); }

int64_t MergerMaskUpdater::getWidth() const { return mergerMaskDefinition->getWidth(); }

int64_t MergerMaskUpdater::getHeight() const { return mergerMaskDefinition->getHeight(); }

std::vector<size_t> MergerMaskUpdater::getMasksOrder() const { return mergerMaskDefinition->getMasksOrder(); }

int MergerMaskUpdater::getInputScaleFactor() const { return mergerMaskDefinition->getInputScaleFactor(); }

std::vector<frameid_t> MergerMaskUpdater::getFrameIds() const { return mergerMaskDefinition->getFrameIds(); }

std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>>
MergerMaskUpdater::getInputIndexPixelDataIfValid(const frameid_t frameId) const {
  return mergerMaskDefinition->getInputIndexPixelDataIfValid(frameId);
}

void MergerMaskUpdater::removeFrameIds(const std::vector<frameid_t>& frameIds) {
  PRESERVE_ACTION(removeFrameIds, mergerMaskDefinition, frameIds);
}

void MergerMaskUpdater::setEnabled(bool b) { PRESERVE_ACTION(setEnabled, mergerMaskDefinition, b); }

void MergerMaskUpdater::setWidth(int64_t int641) { PRESERVE_ACTION(setWidth, mergerMaskDefinition, int641); }

void MergerMaskUpdater::setInputScaleFactor(int scaleFactor) {
  PRESERVE_ACTION(setInputScaleFactor, mergerMaskDefinition, scaleFactor);
}

void MergerMaskUpdater::setHeight(int64_t int641) { PRESERVE_ACTION(setHeight, mergerMaskDefinition, int641); }

void MergerMaskUpdater::setMasksOrder(std::vector<size_t> vector) {
  PRESERVE_ACTION(setMasksOrder, mergerMaskDefinition, vector);
}

const MergerMaskDefinition::InputIndexPixelData& MergerMaskUpdater::getInputIndexPixelData() const {
  return mergerMaskDefinition->getInputIndexPixelData();
}

bool MergerMaskUpdater::validateInputIndexPixelData() const {
  return mergerMaskDefinition->validateInputIndexPixelData();
}

Status MergerMaskUpdater::setInputIndexPixelData(const std::map<videoreaderid_t, std::string>& encodedMasks,
                                                 const uint64_t width, const uint64_t height, const frameid_t frameId) {
  PRESERVE_ACTION_RETURN(setInputIndexPixelData, mergerMaskDefinition, encodedMasks, width, height, frameId);
}

MergerMaskUpdater::MergerMaskUpdater(const MergerMaskDefinition& mergerDefinition)
    : mergerMaskDefinition(mergerDefinition.clone()) {}

}  // namespace Core
}  // namespace VideoStitch
