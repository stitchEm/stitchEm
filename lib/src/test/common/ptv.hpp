// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "fakeReader.hpp"

#include <common/container.hpp>
#include <gpu/core1/transform.hpp>
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/panoDef.hpp"

#include <fstream>
#include <iostream>
#include <atomic>

namespace VideoStitch {
namespace Testing {

/**
 * Creates a Ptv value given a config. Dies on error.
 */
inline Ptv::Value* makePtvValue(const std::string& ptvString) {
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  ENSURE(parser->parseData(ptvString), "Cannot parse ptv data");
  return parser->getRoot().clone();
}

inline Status prepareTransforms(const Core::PanoDefinition& panoDef,
                                std::map<int, std::unique_ptr<Core::Transform>>& transforms) {
  std::atomic<int> readFrameCalls(0), readFrameExits(0);
  std::unique_ptr<FakeReaderFactory> fakeReaderFactory =
      std::unique_ptr<FakeReaderFactory>(new FakeReaderFactory(0, &readFrameCalls, &readFrameExits));
  std::map<readerid_t, std::unique_ptr<Input::VideoReader>> readers;
  for (readerid_t in = 0; in < panoDef.numInputs(); ++in) {
    Potential<Input::Reader> reader = fakeReaderFactory->create(in, panoDef.getInput(in));
    FAIL_RETURN(reader.status());
    Input::VideoReader* videoReader = reader.release()->getVideoReader();
    if (videoReader) {
      readers[in] = std::unique_ptr<Input::VideoReader>(videoReader);
    }
  }
  // Create transforms
  transforms.clear();
  for (auto& reader : readers) {
    const Core::InputDefinition& inputDef = panoDef.getInput(reader.second->id);
    Core::Transform* transform = Core::Transform::create(inputDef);
    if (!transform) {
      return {Origin::Stitcher, ErrType::SetupFailure,
              "Cannot create v1 transformation for input " + std::to_string(reader.second->id)};
    }
    transforms[reader.second->id] = std::unique_ptr<Core::Transform>(transform);
  }

  return Status::OK();
}

inline Status prepareFakeReader(const Core::PanoDefinition& panoDef,
                                std::map<readerid_t, Input::VideoReader*>& readers) {
  deleteAllValues(readers);
  std::atomic<int> readFrameCalls(0), readFrameExits(0);
  std::unique_ptr<FakeReaderFactory> fakeReaderFactory =
      std::unique_ptr<FakeReaderFactory>(new FakeReaderFactory(0, &readFrameCalls, &readFrameExits));
  for (readerid_t in = 0; in < panoDef.numInputs(); ++in) {
    Potential<Input::Reader> reader = fakeReaderFactory->create(in, panoDef.getInput(in));
    FAIL_RETURN(reader.status());
    Input::VideoReader* videoReader = reader.release()->getVideoReader();
    if (videoReader) {
      readers[in] = videoReader;
    }
  }
  return Status::OK();
}

}  // namespace Testing
}  // namespace VideoStitch
