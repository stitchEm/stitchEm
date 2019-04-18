// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "googleauthenticationmanager.hpp"

#include "googlecredentialmodel.hpp"

#include "libvideostitch-base/logmanager.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#endif

#include "google/youtube_api/you_tube_service.h"

#include "googleapis/base/callback-specializations.h"
#include "googleapis/client/auth/oauth2_authorization.h"
#include "googleapis/client/auth/file_credential_store.h"
#include "googleapis/client/data/openssl_codec.h"
#include "googleapis/client/transport/curl_http_transport.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <QCoreApplication>
#include <QInputDialog>
#include <QLabel>

// We use a private implementation to avoid to expose google apis
class GoogleAuthenticationManager::Impl {
  QString storeName = "YouTubeCredentialStore";

 public:
  Impl();
  ~Impl() {}

  googleapis::util::Status initialize();
  googleapis::util::Status authorizeClient(QString newUserName);
  googleapis::util::Status revokeClient(QString userName);
  static googleapis::util::Status promptForAuthorizationCode(googleapis::client::OAuth2AuthorizationFlow* flow,
                                                             const googleapis::client::OAuth2RequestOptions& options,
                                                             std::string* authorization_code);

  std::unique_ptr<googleapis::client::OAuth2AuthorizationFlow> flow;
  std::unique_ptr<googleapis::client::HttpTransportLayerConfig> transportConfig;
  std::shared_ptr<googleapis::client::OAuth2Credential> credential;
  std::shared_ptr<GoogleCredentialModel> credentialModel;
};

GoogleAuthenticationManager::Impl::Impl()
    : transportConfig(new googleapis::client::HttpTransportLayerConfig()),
      credentialModel(new GoogleCredentialModel()) {
  auto factory = new googleapis::client::CurlHttpTransportFactory(transportConfig.get());
  transportConfig->ResetDefaultTransportFactory(factory);
}

googleapis::util::Status GoogleAuthenticationManager::Impl::initialize() {
  std::string secretContent;
  QFile secretFile(":/resources/vahanavr_client_id.json");
  if (secretFile.open(QFile::ReadOnly)) {
    QTextStream stream(&secretFile);
    secretContent = stream.readAll().toStdString();
  } else {
    return googleapis::client::StatusInternalError("No secret file");
  }

  googleapis::util::Status status;
  std::unique_ptr<googleapis::client::HttpTransport> transport(transportConfig->NewDefaultTransport(&status));
  if (!transport || !status.ok()) {
    return status;
  }

  flow.reset(googleapis::client::OAuth2AuthorizationFlow::MakeFlowFromClientSecretsJson(secretContent,
                                                                                        transport.release(), &status));
  if (!status.ok()) {
    return status;
  }

  flow->set_check_email(true);
  flow->set_authorization_code_callback(googleapis::NewPermanentCallback(&promptForAuthorizationCode, flow.get()));

  googleapis::client::OpenSslCodecFactory* openSslFactory = new googleapis::client::OpenSslCodecFactory;
  status = openSslFactory->SetPassphrase(flow->client_spec().client_secret());
  if (status.ok()) {
    std::string homePath;
    status = googleapis::client::FileCredentialStoreFactory::GetSystemHomeDirectoryStorePath(&homePath);
    if (status.ok()) {
      credentialModel->initialize(QString::fromStdString(homePath), storeName);

      googleapis::client::FileCredentialStoreFactory storeFactory(homePath);
      storeFactory.set_codec_factory(openSslFactory);
      flow->ResetCredentialStore(storeFactory.NewCredentialStore(storeName.toStdString(), &status));
    }
  }
  if (!status.ok()) {
    flow.reset();
  }

  return status;
}

googleapis::util::Status GoogleAuthenticationManager::Impl::authorizeClient(QString newUserName) {
  if (credential == nullptr || newUserName != QString::fromStdString(credential->email())) {
    credential.reset(flow->NewCredential());
    googleapis::util::Status status =
        flow->credential_store()->InitCredential(newUserName.toStdString(), credential.get());
    if (status.ok()) {
      credentialModel->internalSetCurrentAndUpdate(newUserName);
      return status;
    } else {  // This is not blocking, just log the error
      QString message = QString::fromStdString(status.ToString());
      VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(message);
    }
  }

  std::vector<string> scopes;
  scopes.push_back(google_youtube_api::YouTubeService::SCOPES::YOUTUBE);

  googleapis::client::OAuth2RequestOptions options;
  options.scopes = googleapis::client::OAuth2AuthorizationFlow::JoinScopes(scopes);
  options.email = newUserName.toStdString();
  googleapis::util::Status status = flow->RefreshCredentialWithOptions(options, credential.get());
  if (status.ok()) {
    credentialModel->internalSetCurrentAndUpdate(newUserName);
  } else {
    credential.reset();
  }
  return status;
}

googleapis::util::Status GoogleAuthenticationManager::Impl::revokeClient(QString userName) {
  flow->credential_store()->Delete(userName.toStdString());
  credentialModel->internalRevokeCredential(userName);
  if (credential && userName == QString::fromStdString(credential->email())) {
    credential.reset();
  }
  return googleapis::client::StatusOk();
}

googleapis::util::Status GoogleAuthenticationManager::Impl::promptForAuthorizationCode(
    googleapis::client::OAuth2AuthorizationFlow* flow, const googleapis::client::OAuth2RequestOptions& options,
    std::string* authorization_code) {
  QString url = QString::fromStdString(flow->GenerateAuthorizationCodeRequestUrlWithOptions(options));
  QString content = QCoreApplication::translate(
      "google auth", "Click %1here%2 to open the authentication page<br/>Please enter the browser's response:");
  QString enhancedContent =
      content.arg("<a href=\"%3\"><span style=\" text-decoration: underline; color:#0000ff;\">").arg("</span></a>");
  QString contentWithUrl =
      enhancedContent.arg(url);  // Add the url after all arguments have been replaced because it contains %n
  QString text = QString("<html><head/><body><p>%1</p></body></html>").arg(contentWithUrl);

  QInputDialog dialog(nullptr, Qt::WindowTitleHint | Qt::WindowSystemMenuHint);
  dialog.setWindowTitle(QCoreApplication::translate("google auth", "Authorization code"));
  dialog.setLabelText(text);
  dialog.findChild<QLabel*>()->setOpenExternalLinks(true);  // after 'setLabelText'

  QString qAuthCode;
  if (dialog.exec()) {
    qAuthCode = dialog.textValue();
  }

  *authorization_code = qAuthCode.toStdString();
  if (authorization_code->empty()) {
    return googleapis::client::StatusCanceled("Canceled");
  } else {
    return googleapis::client::StatusOk();
  }
}

/***************************************************************************/
/*********************** GoogleAuthenticationManager ***********************/
/***************************************************************************/

GoogleAuthenticationManager::GoogleAuthenticationManager() : i(new Impl()) { initialize(); }

GoogleAuthenticationManager::~GoogleAuthenticationManager() {}

bool GoogleAuthenticationManager::initialize() {
  googleapis::util::Status status = i->initialize();
  if (!status.ok()) {
    QString message = QString::fromStdString(status.ToString());
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(message);
  } else {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile("Initialization succeeded");
  }
  return status.ok();
}

bool GoogleAuthenticationManager::authorizeClient(QString userName) {
  googleapis::util::Status status = i->authorizeClient(userName);
  if (!status.ok()) {
    QString message = QString::fromStdString(status.ToString());
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(message);
  } else {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(QString("Authorization succeeded for ") + userName);
  }
  return status.ok();
}

bool GoogleAuthenticationManager::revokeClient(QString userName) {
  googleapis::util::Status status = i->revokeClient(userName);
  if (!status.ok()) {
    QString message = QString::fromStdString(status.ToString());
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(message);
  } else {
    if (authorized() && i->credential->email() == userName.toStdString()) {
      i->credential.reset();
    }
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(QString("Revocation succeeded for ") + userName);
  }
  return status.ok();
}

std::shared_ptr<googleapis::client::OAuth2Credential> GoogleAuthenticationManager::getCredential() const {
  return i->credential;
}

bool GoogleAuthenticationManager::authorized() const { return (bool)i->credential; }

std::shared_ptr<GoogleCredentialModel> GoogleAuthenticationManager::getCredentialModel() const {
  return i->credentialModel;
}
