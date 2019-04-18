// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LIVEEXPOSURE_HPP
#define LIVEEXPOSURE_HPP

#include "libvideostitch/object.hpp"
#include <QString>

/**
 * @brief Class for storing / loading the values of the exposure algorithm configuration
 */
class LiveExposure : public VideoStitch::Ptv::Object {
 public:
  LiveExposure();

  VideoStitch::Ptv::Value* serialize() const;
  const QString getAlgorithm() const;
  int getAnchor() const;
  bool getIsAutoStart() const;
  void setAlgorithm(const QString& algo);
  void setAnchor(const int id);
  void setIsAutoStart(const bool isAuto);

 private:
  QString algorithm;
  int anchor;
  bool autoStart;
};

#endif  // LIVEEXPOSURE_H
