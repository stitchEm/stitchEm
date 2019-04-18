// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QObject>
#include <QtTest>
#include <QApplication>
#include "libvideostitch-gui/utils/smartenum.hpp"

// Generic enumerator for testing
enum class MyEnum { FIRST, SECOND, THIRD, FOURTH };

class MyClass {
 public:
  typedef MyEnum Enum;
  static void initDescriptions(QMap<Enum, QString>& enumToString);
  static const Enum defaultValue = MyEnum::FIRST;
};

class EmptyClass {
 public:
  typedef MyEnum Enum;
  static void initDescriptions(QMap<Enum, QString>& enumToString) { Q_UNUSED(enumToString) }
  static const Enum defaultValue = MyEnum::FIRST;
};

typedef SmartEnum<MyClass, QString> ValueQStringEnum;

typedef SmartEnum<EmptyClass, QString> EmptyEnum;

void MyClass::initDescriptions(QMap<Enum, QString>& enumToString) {
  enumToString[MyEnum::FIRST] = "first";
  enumToString[MyEnum::SECOND] = "second";
  enumToString[MyEnum::THIRD] = "third";
  enumToString[MyEnum::FOURTH] = "fourth";
  ValueQStringEnum::descriptorToEnum["last"] = MyEnum::FOURTH;
}

Q_DECLARE_METATYPE(MyEnum)

class VideostitchTestSmartEnums : public QObject {
  Q_OBJECT

 public:
  VideostitchTestSmartEnums();

 private Q_SLOTS:
  // Test equality between enums, strings and smart enums
  void testEmptyEnum();
  void testEqual();
  void testDefaultDescriptor();
  void testDescriptorList();
  void testEqual_data();
  // Following test modify the SmartEnum ValueQStringEnum
  void testDescriptorExtension();
};

VideostitchTestSmartEnums::VideostitchTestSmartEnums() {}

void VideostitchTestSmartEnums::testEmptyEnum() {
  // Empty case should return the default value
  QCOMPARE(EmptyEnum::getEnumFromDescriptor("test").getValue(), MyEnum::FIRST);
  QCOMPARE(EmptyEnum::getDefaultDescriptor(), QStringLiteral(""));
}

void VideostitchTestSmartEnums::testEqual_data() {
  QTest::addColumn<QString>("descriptor");
  QTest::addColumn<MyEnum>("enumerator");
  QTest::newRow("first value") << "first" << MyEnum::FIRST;
  QTest::newRow("second value") << "second" << MyEnum::SECOND;
  QTest::newRow("third value") << "third" << MyEnum::THIRD;
  QTest::newRow("fourth value") << "fourth" << MyEnum::FOURTH;
}

void VideostitchTestSmartEnums::testEqual() {
  QFETCH(QString, descriptor);
  QFETCH(MyEnum, enumerator);
  QCOMPARE(ValueQStringEnum::getEnumFromDescriptor(ValueQStringEnum::getDescriptorFromEnum(enumerator)).getValue(),
           enumerator);
  QCOMPARE(ValueQStringEnum::getDescriptorFromEnum(ValueQStringEnum::getEnumFromDescriptor(descriptor).getValue()),
           descriptor);
  QCOMPARE(ValueQStringEnum::getEnumFromDescriptor(ValueQStringEnum::getDescriptorFromEnum(enumerator)).getDescriptor(),
           descriptor);
  QCOMPARE(ValueQStringEnum::getValueFromDescriptor(descriptor), enumerator);
  QVERIFY(ValueQStringEnum::getEnumFromDescriptor(descriptor) == enumerator);
}

void VideostitchTestSmartEnums::testDefaultDescriptor() {
  QVERIFY(ValueQStringEnum::getDefaultDescriptor() == QStringLiteral("first"));
  QVERIFY(ValueQStringEnum::getDefaultDescriptor() == ValueQStringEnum::getDescriptorFromEnum(MyEnum::FIRST));
}

void VideostitchTestSmartEnums::testDescriptorList() { QVERIFY(ValueQStringEnum::getDescriptorsList().size() > 0); }

void VideostitchTestSmartEnums::testDescriptorExtension() {
  // duplicated descriptor for the same enum (within initDescriptions)
  QCOMPARE(ValueQStringEnum::getValueFromDescriptor("fourth"), MyEnum::FOURTH);
  QCOMPARE(ValueQStringEnum::getValueFromDescriptor("last"), MyEnum::FOURTH);
  // additional descriptors for the same enum (after initialization)
  // keep initial descriptor in the list but change the corresponding descriptor
  ValueQStringEnum::setDescriptorForEnum(MyEnum::SECOND, "next");
  QCOMPARE(ValueQStringEnum::getValueFromDescriptor("second"), MyEnum::SECOND);
  QCOMPARE(ValueQStringEnum::getValueFromDescriptor("next"), MyEnum::SECOND);
  QCOMPARE(ValueQStringEnum::getDescriptorFromEnum(MyEnum::SECOND), QStringLiteral("next"));
}

QTEST_APPLESS_MAIN(VideostitchTestSmartEnums)

#include "smartEnumTest.moc"
