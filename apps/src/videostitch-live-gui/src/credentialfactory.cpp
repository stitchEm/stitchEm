// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "credentialfactory.hpp"

#include "googleauthenticationmanager.hpp"
#include "googlecredentialmodel.hpp"

std::shared_ptr<CredentialModel> CredentialFactory::getCredentialModel(OutputCredential outputCredential) {
#ifdef ENABLE_YOUTUBE_OUTPUT
  switch (outputCredential) {
    case OutputCredential::YouTube:
      return GoogleAuthenticationManager::getInstance().getCredentialModel();
    default:
      return nullptr;
  }
#else   // ENABLE_YOUTUBE_OUTPUT
  Q_UNUSED(outputCredential);
  return nullptr;
#endif  // ENABLE_YOUTUBE_OUTPUT
}
