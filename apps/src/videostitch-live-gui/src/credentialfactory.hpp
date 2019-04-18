// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <memory>

enum class OutputCredential { YouTube };

class CredentialModel;

class CredentialFactory {
 public:
  static std::shared_ptr<CredentialModel> getCredentialModel(OutputCredential outputCredential);
};
