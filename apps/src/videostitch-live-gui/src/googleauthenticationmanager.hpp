// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <memory>
#include <string>

namespace googleapis {
namespace client {
class OAuth2Credential;
}
}  // namespace googleapis

class GoogleCredentialModel;

class GoogleAuthenticationManager {
 public:
  static GoogleAuthenticationManager& getInstance() {
    static GoogleAuthenticationManager instance;

    return instance;
  }

  ~GoogleAuthenticationManager();

  bool authorizeClient(QString userName);
  bool revokeClient(QString userName);

  std::shared_ptr<googleapis::client::OAuth2Credential> getCredential() const;
  bool authorized() const;
  std::shared_ptr<GoogleCredentialModel> getCredentialModel() const;  // Keeps ownership

 private:
  GoogleAuthenticationManager();
  GoogleAuthenticationManager(const GoogleAuthenticationManager&) = delete;
  void operator=(const GoogleAuthenticationManager&) = delete;
  bool initialize();

  class Impl;
  std::unique_ptr<Impl> i;
};
