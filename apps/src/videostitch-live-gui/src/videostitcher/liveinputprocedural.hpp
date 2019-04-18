// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LIVEINPUTPROCEDURAL_HPP
#define LIVEINPUTPROCEDURAL_HPP

#include "liveinputfactory.hpp"

#include <QColor>

namespace Procedural {
enum Name { frameNumber, grid, color };

class NameClass {
 public:
  typedef Name Enum;
  static void initDescriptions(QMap<Enum, QString>& enumToString);
  static const Enum defaultValue;
};

typedef SmartEnum<NameClass, QString> NameEnum;
}  // namespace Procedural

class LiveInputProcedural : public LiveInputFactory {
 public:
  explicit LiveInputProcedural(const QString& name);
  ~LiveInputProcedural();

  virtual VideoStitch::Ptv::Value* serialize() const;
  virtual void initializeWith(const VideoStitch::Ptv::Value* initializationInput);
  virtual VideoStitch::InputFormat::InputFormatEnum getType() const;

  int getWidth() const;
  int getHeight() const;
  QColor getColor() const;

  void setWidth(int newWidth);
  void setHeight(int newHeight);
  void setColor(QColor newColor);

 private:
  int width;
  int height;
  QColor color;
};

#endif  // LIVEINPUTPROCEDURAL_H
