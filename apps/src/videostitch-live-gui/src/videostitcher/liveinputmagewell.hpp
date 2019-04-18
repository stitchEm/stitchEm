// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LIVEINPUTMAGEWELL_HPP
#define LIVEINPUTMAGEWELL_HPP

#include "capturecardliveinput.hpp"

namespace Magewell {
enum BuiltInZoom { Zoom, Fill, None };

class BuiltInZoomClass {
 public:
  typedef BuiltInZoom Enum;
  static void initDescriptions(QMap<Enum, QString>& enumToString);
  static const Enum defaultValue;
};

typedef SmartEnum<BuiltInZoomClass, QString> BuiltInZoomEnum;
}  // namespace Magewell

class LiveInputMagewell : public CaptureCardLiveInput {
 public:
  explicit LiveInputMagewell(const QString& name);
  ~LiveInputMagewell();

  virtual VideoStitch::Ptv::Value* serialize() const;
  virtual void initializeWith(const VideoStitch::Ptv::Value* initializationInput);
  virtual VideoStitch::InputFormat::InputFormatEnum getType() const;

  Magewell::BuiltInZoomEnum getBuiltInZoom() const;
  void setBuildInZoom(const Magewell::BuiltInZoomEnum& newBuiltInZoom);

 private:
  Magewell::BuiltInZoomEnum builtInZoom;
};

class LiveInputMagewellPro : public LiveInputMagewell {
 public:
  explicit LiveInputMagewellPro(const QString& name);
  ~LiveInputMagewellPro();
  virtual VideoStitch::InputFormat::InputFormatEnum getType() const;
};

#endif  // LIVEINPUTMAGEWELL_HPP
