// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveoutputlist.hpp"

#include "liveoutputfactory.hpp"

#include "libvideostitch/status.hpp"

LiveOutputList::LiveOutputList() {}

void LiveOutputList::addOutput(LiveOutputFactory* output) {
  if (output != nullptr) {
    map[output->getIdentifier()] = output;
  }
}

LiveOutputFactory* LiveOutputList::getOutput(const QString& id) const { return map.value(id, nullptr); }

void LiveOutputList::removeOutput(const QString& id) { map.remove(id); }

void LiveOutputList::replaceOutput(const QString& existingOne) {
  LiveOutputFactory* output = getOutput(existingOne);
  removeOutput(existingOne);
  addOutput(output);
}

void LiveOutputList::clearOutput() {
  for (auto liveOutput : map.values()) {
    delete liveOutput;
  }
  map.clear();
}

bool LiveOutputList::isEmpty() const { return map.empty(); }

unsigned int LiveOutputList::activeOutputs() const {
  unsigned int actives = 0u;
  for (auto output : map.values()) {
    if (output->getOutputState() == LiveOutputFactory::OutputState::ENABLED ||
        output->getOutputState() == LiveOutputFactory::OutputState::CONNECTING ||
        output->getOutputState() == LiveOutputFactory::OutputState::INITIALIZATION)
      actives++;
  }
  return actives;
}

QList<LiveOutputFactory*> LiveOutputList::getValues() const {
  // TODO: Provide an interator instead of returning the whole map
  return map.values();
}
