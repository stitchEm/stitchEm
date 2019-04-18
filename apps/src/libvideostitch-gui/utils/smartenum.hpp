// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef SMARTENUM_HPP
#define SMARTENUM_HPP

#include <QMap>

/* Mutable enumerator */
template <typename Enumerator, typename Descriptor>
class SmartEnum {
  friend Enumerator;

 public:
  typedef typename Enumerator::Enum Enum;

  static SmartEnum<Enumerator, Descriptor> getEnumFromDescriptor(const Descriptor& description);

  static Descriptor getDescriptorFromEnum(const Enum& value);

  static Enum getValueFromDescriptor(const Descriptor& description);

  static void setDescriptorForEnum(const Enum& value, const Descriptor& description);

  static QList<Descriptor> getDescriptorsList();

  static Descriptor getDefaultDescriptor();

  explicit SmartEnum(const Enum& value);

  SmartEnum();

  inline Enum getValue() const;

  inline Descriptor getDescriptor() const;

  static typename SmartEnum<Enumerator, Descriptor>::Enum defaultValue;

  bool operator==(const Enum) const;
  bool operator!=(const Enum) const;

 protected:
  Enum value;

  Descriptor descriptor;

  static QMap<typename SmartEnum<Enumerator, Descriptor>::Enum, Descriptor> enumToDescriptor;

  static QMap<Descriptor, typename SmartEnum<Enumerator, Descriptor>::Enum> descriptorToEnum;

  static void init();
};

template <typename Enumerator, typename Descriptor>
SmartEnum<Enumerator, Descriptor> SmartEnum<Enumerator, Descriptor>::getEnumFromDescriptor(
    const Descriptor& descriptor) {
  if (descriptorToEnum.empty()) {
    init();
  }
  if (descriptorToEnum.count(descriptor) > 0) {
    return SmartEnum<Enumerator, Descriptor>(descriptorToEnum[descriptor]);
  }
  return SmartEnum<Enumerator, Descriptor>(defaultValue);
}

template <typename Enumerator, typename Descriptor>
Descriptor SmartEnum<Enumerator, Descriptor>::getDescriptorFromEnum(
    const typename SmartEnum<Enumerator, Descriptor>::Enum& value) {
  if (enumToDescriptor.empty()) {
    init();
  }
  if (enumToDescriptor.contains(value)) {
    return enumToDescriptor.value(value);
  }
  return getDefaultDescriptor();
}

template <typename Enumerator, typename Descriptor>
typename SmartEnum<Enumerator, Descriptor>::Enum SmartEnum<Enumerator, Descriptor>::getValueFromDescriptor(
    const Descriptor& descriptor) {
  if (descriptorToEnum.empty()) {
    init();
  }
  if (descriptorToEnum.count(descriptor) > 0) {
    return descriptorToEnum[descriptor];
  }
  return defaultValue;
}

template <typename Enumerator, typename Descriptor>
void SmartEnum<Enumerator, Descriptor>::setDescriptorForEnum(
    const typename SmartEnum<Enumerator, Descriptor>::Enum& value, const Descriptor& description) {
  if (enumToDescriptor.empty()) {
    init();
  }

  enumToDescriptor[value] = description;
  descriptorToEnum[description] = value;
}

template <typename Enumerator, typename Descriptor>
QList<Descriptor> SmartEnum<Enumerator, Descriptor>::getDescriptorsList() {
  if (enumToDescriptor.empty()) {
    init();
  }
  QList<Descriptor> list = descriptorToEnum.keys();

  return list;
}

template <typename Enumerator, typename Descriptor>
Descriptor SmartEnum<Enumerator, Descriptor>::getDefaultDescriptor() {
  if (enumToDescriptor.empty()) {
    init();
  }
  return enumToDescriptor.value(defaultValue);
}

template <typename Enumerator, typename Descriptor>
SmartEnum<Enumerator, Descriptor>::SmartEnum(const typename SmartEnum<Enumerator, Descriptor>::Enum& value)
    : value(value), descriptor(getDescriptorFromEnum(value)) {}

template <typename Enumerator, typename Descriptor>
SmartEnum<Enumerator, Descriptor>::SmartEnum() : value(defaultValue), descriptor(getDescriptorFromEnum(defaultValue)) {}

template <typename Enumerator, typename Descriptor>
inline typename SmartEnum<Enumerator, Descriptor>::Enum SmartEnum<Enumerator, Descriptor>::getValue() const {
  return value;
}

template <typename Enumerator, typename Descriptor>
inline Descriptor SmartEnum<Enumerator, Descriptor>::getDescriptor() const {
  return descriptor;
}

template <typename Enumerator, typename Descriptor>
inline bool SmartEnum<Enumerator, Descriptor>::operator==(const Enum enumerator) const {
  return value == enumerator;
}

template <typename Enumerator, typename Descriptor>
inline bool SmartEnum<Enumerator, Descriptor>::operator!=(const Enum enumerator) const {
  return !operator==(enumerator);
}

template <typename Enumerator, typename Descriptor>
typename SmartEnum<Enumerator, Descriptor>::Enum SmartEnum<Enumerator, Descriptor>::defaultValue =
    (typename SmartEnum<Enumerator, Descriptor>::Enum)(-1);

template <typename Enumerator, typename Descriptor>
QMap<typename SmartEnum<Enumerator, Descriptor>::Enum, Descriptor> SmartEnum<Enumerator, Descriptor>::enumToDescriptor;

template <typename Enumerator, typename Descriptor>
QMap<Descriptor, typename SmartEnum<Enumerator, Descriptor>::Enum> SmartEnum<Enumerator, Descriptor>::descriptorToEnum;

template <typename Enumerator, typename Descriptor>
void SmartEnum<Enumerator, Descriptor>::init() {
  Enumerator::initDescriptions(enumToDescriptor);
  typename QMap<Enum, Descriptor>::iterator it;

  for (it = enumToDescriptor.begin(); it != enumToDescriptor.end(); ++it) {
    descriptorToEnum[it.value()] = it.key();
  }
  SmartEnum<Enumerator, Descriptor>::defaultValue = Enumerator::defaultValue;
}

#endif  // SMARTENUM_HPP
