// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/algorithm.hpp"

#include "registeredAlgo.hpp"

namespace VideoStitch {
namespace Util {
Algorithm::Algorithm() {}

Algorithm::~Algorithm() {}

void Algorithm::list(std::vector<std::string>& algos) {
  algos.clear();
  typedef RegisteredAlgoBase<false>::InstanceMap AlgoMap;
  for (const AlgoMap::value_type& inst : RegisteredAlgoBase<false>::getInstances()) {
    algos.push_back(inst.first);
  }
}

const char* Algorithm::getDocString(const std::string& name) {
  RegisteredAlgoBase<false>* reg = RegisteredAlgoBase<false>::getInstance(name);
  if (reg) {
    return reg->getDocString();
  }
  return nullptr;
}

Potential<Algorithm> Algorithm::create(const std::string& name, const Ptv::Value* config) {
  RegisteredAlgoBase<false>* reg = RegisteredAlgoBase<false>::getInstance(name);
  if (reg) {
    return reg->create(config);
  }
  return {Origin::PanoramaConfiguration, ErrType::UnsupportedAction, "Unknown algorithm: '" + name + "'"};
}

OnlineAlgorithm::OnlineAlgorithm() {}

OnlineAlgorithm::~OnlineAlgorithm() {}

void OnlineAlgorithm::list(std::vector<std::string>& algos) {
  algos.clear();
  typedef RegisteredAlgoBase<true>::InstanceMap AlgoMap;
  for (const AlgoMap::value_type& inst : RegisteredAlgoBase<true>::getInstances()) {
    algos.push_back(inst.first);
  }
}

const char* OnlineAlgorithm::getDocString(const std::string& name) {
  RegisteredAlgoBase<false>* reg = RegisteredAlgoBase<false>::getInstance(name);
  if (reg) {
    return reg->getDocString();
  }
  return nullptr;
}

Potential<OnlineAlgorithm> OnlineAlgorithm::create(const std::string& name, const Ptv::Value* config) {
  RegisteredAlgoBase<true>* reg = RegisteredAlgoBase<true>::getInstance(name);
  if (reg) {
    return reg->create(config);
  }
  return {Origin::PanoramaConfiguration, ErrType::UnsupportedAction, "Unknown algorithm: '" + name + "'"};
}

}  // namespace Util
}  // namespace VideoStitch
