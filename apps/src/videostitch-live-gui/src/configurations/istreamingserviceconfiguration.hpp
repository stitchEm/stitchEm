// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef ISTREAMINGSERVICECONFIGURATION_HPP
#define ISTREAMINGSERVICECONFIGURATION_HPP

#include <QGroupBox>

class LiveOutputRTMP;
class LiveProjectDefinition;

class IStreamingServiceConfiguration : public QGroupBox {
  Q_OBJECT
 public:
  static IStreamingServiceConfiguration* createStreamingService(QWidget* parent = nullptr,
                                                                LiveOutputRTMP* outputRef = nullptr,
                                                                LiveProjectDefinition* projectDefinition = nullptr);

  explicit IStreamingServiceConfiguration(QWidget* parent = nullptr, LiveOutputRTMP* outputRef = nullptr,
                                          LiveProjectDefinition* projectDefinition = nullptr);

  virtual bool loadConfiguration() = 0;
  virtual void saveConfiguration() = 0;
  virtual bool hasValidConfiguration() const { return true; }
  virtual void startBaseConfiguration() {}

  void setLiveProjectDefinition(LiveProjectDefinition* value);

 signals:
  void stateChanged();
  void basicConfigurationComplete();
  void basicConfigurationCanceled();

 protected:
  LiveOutputRTMP* outputRef;
  LiveProjectDefinition* liveProjectDefinition;
};

#endif  // ISTREAMINGSERVICECONFIGURATION_HPP
