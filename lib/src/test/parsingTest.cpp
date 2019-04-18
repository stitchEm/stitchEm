// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <parse/json.hpp>
#include <parse/jsonDriver.hpp>
#include <util/strutils.hpp>
#include "libvideostitch/logging.hpp"
#include "libvideostitch/ptv.hpp"

#include <cstdint>
#include <limits>
#include <fstream>

namespace VideoStitch {
namespace Testing {

#ifdef _MSC_VER
// bug VSA-6947
// 64 Bit integers are truncated on Windows
using MAX_SERIALIZABLE_INT_TYPE = int32_t;
#else
using MAX_SERIALIZABLE_INT_TYPE = int64_t;
#endif

static const MAX_SERIALIZABLE_INT_TYPE BIG_NEG_INT = std::numeric_limits<MAX_SERIALIZABLE_INT_TYPE>::min();
static const MAX_SERIALIZABLE_INT_TYPE BIG_POS_INT = std::numeric_limits<MAX_SERIALIZABLE_INT_TYPE>::max();

#define TMPFILE "__tmp__test__.ptv"
void testPopulate() {
  {
    std::ofstream ofs(TMPFILE, std::ios_base::out);
    ofs << R"(
    {
      "b": true,
      "i": 1,
      "f": 3.14,
      "c1": "ee12f3",
      "c2": "fe12f3ff",
      "ce": "e12f3",
      "l1": [],
      "l2": [-1, 0, 1, )";
    ofs << BIG_NEG_INT << "," << BIG_POS_INT << "],";
    ofs << "\"i64\": " << BIG_POS_INT << "}";
  }
  Parse::JsonDriver driver(false, false);
  ENSURE(driver.parse(TMPFILE));

  // Disable error messages.
  Logger::LogLevel backupLevel = Logger::getLevel();
  Logger::setLevel(Logger::Quiet);

  {
    bool b = false;
    ENSURE(Parse::populateBool("Blah", driver.getRoot(), "b", b, true) == Parse::PopulateResult_Ok);
    ENSURE_EQ(b, true);
    b = false;
    ENSURE(Parse::populateBool("Blah", driver.getRoot(), "b", b, false) == Parse::PopulateResult_Ok);
    ENSURE_EQ(b, true);
    b = false;
    // Int is convertible to bool.
    ENSURE(Parse::populateBool("Blah", driver.getRoot(), "i", b, false) == Parse::PopulateResult_Ok);
    ENSURE_EQ(b, true);
    // Others are not.
    ENSURE(Parse::populateBool("Blah", driver.getRoot(), "f", b, false) == Parse::PopulateResult_WrongType);
    ENSURE(Parse::populateBool("Blah", driver.getRoot(), "c1", b, false) == Parse::PopulateResult_WrongType);
    ENSURE(Parse::populateBool("Blah", driver.getRoot(), "notthere", b, false) == Parse::PopulateResult_DoesNotExist);
  }

  {
    MAX_SERIALIZABLE_INT_TYPE i = 0;
    MAX_SERIALIZABLE_INT_TYPE one = 1;
    ENSURE(Parse::populateInt("Blah", driver.getRoot(), "i", i, true) == Parse::PopulateResult_Ok);
    ENSURE_EQ(i, one);
    i = 0;
    ENSURE(Parse::populateInt("Blah", driver.getRoot(), "i64", i, true) == Parse::PopulateResult_Ok);
    ENSURE_EQ(i, BIG_POS_INT);
    i = 0;
    ENSURE(Parse::populateInt("Blah", driver.getRoot(), "i", i, false) == Parse::PopulateResult_Ok);
    ENSURE_EQ(i, one);
    i = 0;
    ENSURE(Parse::populateInt("Blah", driver.getRoot(), "i64", i, false) == Parse::PopulateResult_Ok);
    ENSURE_EQ(i, BIG_POS_INT);
    i = 0;
    ENSURE(Parse::populateInt("Blah", driver.getRoot(), "b", i, false) == Parse::PopulateResult_WrongType);
    ENSURE(Parse::populateInt("Blah", driver.getRoot(), "f", i, false) == Parse::PopulateResult_WrongType);
    ENSURE(Parse::populateInt("Blah", driver.getRoot(), "c1", i, false) == Parse::PopulateResult_WrongType);
    ENSURE(Parse::populateInt("Blah", driver.getRoot(), "notthere", i, false) == Parse::PopulateResult_DoesNotExist);
  }

  {
    double f = 0.0;
    ENSURE(Parse::populateDouble("Blah", driver.getRoot(), "f", f, true) == Parse::PopulateResult_Ok);
    ENSURE_APPROX_EQ(f, 3.14, 0.01);
    f = 0.0;
    ENSURE(Parse::populateDouble("Blah", driver.getRoot(), "f", f, false) == Parse::PopulateResult_Ok);
    ENSURE_APPROX_EQ(f, 3.14, 0.01);
    f = 0.0;
    // Int is convertible to float.
    ENSURE(Parse::populateDouble("Blah", driver.getRoot(), "i", f, false) == Parse::PopulateResult_Ok);
    ENSURE_APPROX_EQ(f, 1.0, 0.01);
    // Others are not.
    ENSURE(Parse::populateDouble("Blah", driver.getRoot(), "b", f, false) == Parse::PopulateResult_WrongType);
    ENSURE(Parse::populateDouble("Blah", driver.getRoot(), "c1", f, false) == Parse::PopulateResult_WrongType);
    ENSURE(Parse::populateDouble("Blah", driver.getRoot(), "notthere", f, false) == Parse::PopulateResult_DoesNotExist);
  }

  {
    std::string s;
    ENSURE(Parse::populateString("Blah", driver.getRoot(), "c1", s, true) == Parse::PopulateResult_Ok);
    ENSURE_EQ(s, std::string("ee12f3"));
    s.clear();
    ENSURE(Parse::populateString("Blah", driver.getRoot(), "c1", s, false) == Parse::PopulateResult_Ok);
    ENSURE_EQ(s, std::string("ee12f3"));
    s.clear();
    ENSURE(Parse::populateString("Blah", driver.getRoot(), "b", s, false) == Parse::PopulateResult_WrongType);
    ENSURE(Parse::populateString("Blah", driver.getRoot(), "i", s, false) == Parse::PopulateResult_WrongType);
    ENSURE(Parse::populateString("Blah", driver.getRoot(), "notthere", s, false) == Parse::PopulateResult_DoesNotExist);
  }

  {
    uint32_t c = 0;
    ENSURE(Parse::populateColor("Blah", driver.getRoot(), "c1", c, true) == Parse::PopulateResult_Ok);
    ENSURE_EQ(c, (uint32_t)0xfff312eeU);  // ABGR
    ENSURE(Parse::populateColor("Blah", driver.getRoot(), "c1", c, false) == Parse::PopulateResult_Ok);
    ENSURE_EQ(c, (uint32_t)0xfff312eeU);
    ENSURE(Parse::populateColor("Blah", driver.getRoot(), "c2", c, true) == Parse::PopulateResult_Ok);
    ENSURE_EQ(c, (uint32_t)0xfff312feU);
    ENSURE(Parse::populateColor("Blah", driver.getRoot(), "c2", c, false) == Parse::PopulateResult_Ok);
    ENSURE_EQ(c, (uint32_t)0xfff312feU);
    ENSURE(Parse::populateColor("Blah", driver.getRoot(), "b", c, false) == Parse::PopulateResult_WrongType);
    ENSURE(Parse::populateColor("Blah", driver.getRoot(), "i", c, false) == Parse::PopulateResult_WrongType);
    ENSURE(Parse::populateColor("Blah", driver.getRoot(), "ce", c, false) == Parse::PopulateResult_WrongType);
    ENSURE(Parse::populateColor("Blah", driver.getRoot(), "notthere", c, false) == Parse::PopulateResult_DoesNotExist);
  }

  {
    {
      std::vector<int64_t> v;
      ENSURE(Parse::populateIntList("Blah", driver.getRoot(), "l1", v, true) == Parse::PopulateResult_Ok);
      ENSURE(v.empty());
    }

    {
      std::vector<int64_t> v;
      ENSURE(Parse::populateIntList("Blah", driver.getRoot(), "l2", v, true) == Parse::PopulateResult_Ok);
      ENSURE_EQ(v, std::vector<int64_t>{-1, 0, 1, BIG_NEG_INT, BIG_POS_INT});
    }

    {
      std::vector<int64_t> v;
      ENSURE(Parse::populateIntList("Blah", driver.getRoot(), "l1", v, false) == Parse::PopulateResult_Ok);
      ENSURE(v.empty());
    }

    {
      std::vector<int64_t> v;
      ENSURE(Parse::populateIntList("Blah", driver.getRoot(), "l2", v, false) == Parse::PopulateResult_Ok);
      ENSURE_EQ(v, std::vector<int64_t>{-1, 0, 1, BIG_NEG_INT, BIG_POS_INT});
    }

    {
      std::vector<int64_t> v;
      ENSURE(Parse::populateIntList("Blah", driver.getRoot(), "b", v, false) == Parse::PopulateResult_WrongType);
      ENSURE(Parse::populateIntList("Blah", driver.getRoot(), "i", v, false) == Parse::PopulateResult_WrongType);
      ENSURE(Parse::populateIntList("Blah", driver.getRoot(), "ce", v, false) == Parse::PopulateResult_WrongType);
      ENSURE(Parse::populateIntList("Blah", driver.getRoot(), "c1", v, false) == Parse::PopulateResult_WrongType);
      ENSURE(Parse::populateIntList("Blah", driver.getRoot(), "c2", v, false) == Parse::PopulateResult_WrongType);
      ENSURE(Parse::populateIntList("Blah", driver.getRoot(), "notthere", v, false) ==
             Parse::PopulateResult_DoesNotExist);
    }
  }

  // Restore error messages.
  Logger::setLevel(backupLevel);
}

void testNoFile() {
  Parse::JsonDriver driver(false, false);
  ENSURE(!driver.parse("gzerhgarh"), "File does not exist, should have failed.");
}

void testParse1() {
  Parse::JsonDriver driver(false, false);
  ENSURE(driver.parse("data/simple1_json.ptv"), driver.getErrorMessage().c_str());

  ENSURE(driver.getRoot().has("doesnotexist") == NULL);
  ENSURE(driver.getRoot().has("a") != NULL);
  ENSURE_EQ((int64_t)1, driver.getRoot().has("a")->asInt());
  ENSURE_EQ(2.1, driver.getRoot().has("b")->asDouble());
  ENSURE_EQ(std::string("to\t\"\\to"), driver.getRoot().has("c")->asString());
  ENSURE_EQ(std::string("3.2"), driver.getRoot().has("dodo")->asString());
  ENSURE_EQ(Ptv::Value::NIL, driver.getRoot().has("e")->getType());
  ENSURE_EQ(true, driver.getRoot().has("f")->asBool());
  ENSURE_EQ(false, driver.getRoot().has("g")->asBool());

  ENSURE(driver.getRoot() == driver.getRoot());

  ENSURE_EQ(Ptv::Value::LIST, driver.getRoot().has("h")->getType());
  {
    const std::vector<Ptv::Value*>& l = driver.getRoot().has("h")->asList();
    ENSURE_EQ(4, (int)l.size());
    ENSURE_EQ(Ptv::Value::OBJECT, l[0]->getType());
    ENSURE_EQ(Ptv::Value::LIST, l[1]->getType());
    ENSURE_EQ(Ptv::Value::OBJECT, l[2]->getType());
    ENSURE_EQ(Ptv::Value::INT, l[3]->getType());
    ENSURE_EQ((int64_t)1, l[3]->asInt());
  }

  // Make sure copy works.
  Ptv::Value* rootCopy = driver.getRoot().clone();
  ENSURE_EQ(Ptv::Value::OBJECT, rootCopy->getType());

  ENSURE(rootCopy->has("a") != NULL);
  ENSURE_EQ((int64_t)1, rootCopy->has("a")->asInt());
  ENSURE_EQ(2.1, rootCopy->has("b")->asDouble());
  ENSURE_EQ(std::string("to\t\"\\to"), rootCopy->has("c")->asString());
  ENSURE_EQ(std::string("3.2"), rootCopy->has("dodo")->asString());
  ENSURE_EQ(Ptv::Value::NIL, rootCopy->has("e")->getType());
  ENSURE_EQ(true, rootCopy->has("f")->asBool());
  ENSURE_EQ(false, rootCopy->has("g")->asBool());

  ENSURE_EQ(Ptv::Value::LIST, rootCopy->has("h")->getType());
  {
    const std::vector<Ptv::Value*>& l = rootCopy->has("h")->asList();
    ENSURE_EQ(4, (int)l.size());
    ENSURE_EQ(Ptv::Value::OBJECT, l[0]->getType());
    ENSURE_EQ(Ptv::Value::LIST, l[1]->getType());
    ENSURE_EQ(Ptv::Value::OBJECT, l[2]->getType());
    ENSURE_EQ(Ptv::Value::INT, l[3]->getType());
    ENSURE_EQ((int64_t)1, l[3]->asInt());
  }

  // Make sure order is consistent.
  delete rootCopy->remove("a");
  rootCopy->get("a")->asInt() = 42;

  const std::string expected(
      "{\n"
      "  \"b\" : 2.1000000000000001, \n"
      "  \"c\" : \"to\\t\\\"\\\\to\", \n"
      "  \"dodo\" : \"3.2\", \n"
      "  \"e\" : null, \n"
      "  \"f\" : true, \n"
      "  \"g\" : false, \n"
      "  \"h\" : [\n"
      "    {\n"
      "      \"ha\" : 1\n"
      "    },\n"
      "    [],\n"
      "    {},\n"
      "    1\n"
      "  ], \n"
      "  \"a\" : 42\n"
      "}");

  {
    std::stringstream ss;
    rootCopy->printJson(ss);
    ENSURE_EQ(expected, ss.str());
  }

  delete rootCopy;
}

void testEscapeFail() {
  Parse::JsonDriver driver(false, false);
  ENSURE(!driver.parse("data/escapeFail_json.ptv"));
  // ENSURE(driver.getErrorMessage().find("invalid escape sequence") != std::string::npos,
  // driver.getErrorMessage().c_str());
}

void testParseWithPano() {
  Parse::JsonDriver driver(false, false);
  ENSURE(driver.parse("data/pano_json.ptv"));
  // driver.getRoot().printJson(std::cout);
  ENSURE(driver.getRoot() == driver.getRoot());
}

void testDefaults(const char* input, const char* defaults, const char* expected) {
  {
    std::ofstream ofs(TMPFILE, std::ios_base::out);
    ofs << input;
  }
  Parse::JsonDriver driver(false, false);
  ENSURE(driver.parse(TMPFILE));
  std::unique_ptr<Ptv::Value> ptv(driver.getRoot().clone());
  {
    std::ofstream ofs(TMPFILE, std::ios_base::out);
    ofs << defaults;
  }
  ENSURE(driver.parse(TMPFILE));

  ptv->populateWithPrimitiveDefaults(driver.getRoot());

  {
    std::ofstream ofs(TMPFILE, std::ios_base::out);
    ofs << expected;
  }
  ENSURE(driver.parse(TMPFILE));

  ENSURE(driver.getRoot() == *ptv);
}

void testOrderedMap() {
#define TEST_ORDERED_MAP_SIZE 16

  std::vector<int*> t(TEST_ORDERED_MAP_SIZE);
  int first = 0, last = TEST_ORDERED_MAP_SIZE;
  Parse::OrderedMap<int> map;
  for (int i = first; i < last; ++i) {
    t[i] = new int(i);
    std::stringstream ss;
    ss << "test " << i;
    ENSURE(map.put(ss.str(), t[i]));
  }

  // test order
  for (int i = first; i < last; ++i) {
    std::stringstream ss;
    ss << "test " << i;
    ENSURE_EQ(*map.get(i).first, ss.str());
    ENSURE_EQ(*map.get(i).second, i);
  }

  // reverse
  map.reverse();
  for (int i = first; i < last; ++i) {
    int inv = last - first - 1 - i;
    std::stringstream ss;
    ss << "test " << inv;
    ENSURE_EQ(*map.get(i).first, ss.str());
    ENSURE_EQ(*map.get(i).second, inv);
  }
  map.reverse();

  // remove/add twice
  {
    std::stringstream ss;
    size_t i = map.size() - 1;
    ss << "test " << i;
    ENSURE_EQ(*map.remove(ss.str()), (int)i);
    ENSURE(map.remove(ss.str()) == NULL);
    ENSURE(map.put(ss.str(), t[i]));
    ENSURE(!map.put(ss.str(), t[i]));
  }

  // remove + compact: head, tail
  {
    int* entry0 = map.remove("test 0");
    ENSURE_EQ(*entry0, 0);
    delete entry0;
    first++;
    std::stringstream ss;
    size_t i = map.size();
    ss << "test " << i;
    int* lastEntry = map.remove(ss.str());
    ENSURE_EQ(*lastEntry, (int)i);
    delete lastEntry;
    last--;
    // test order
    for (int i = first - 1, j = first; j < last; ++i, ++j) {
      std::stringstream ss;
      ss << "test " << j;
      ENSURE_EQ(*map.get(i).first, ss.str());
      ENSURE_EQ(*map.get(i).second, j);
    }
    map.compact();
    // test order
    for (int i = first - 1, j = first; j < last; ++i, ++j) {
      std::stringstream ss;
      ss << "test " << j;
      ENSURE_EQ(*map.get(i).first, ss.str());
      ENSURE_EQ(*map.get(i).second, j);
    }
    ENSURE_EQ((int)map.size(), last - first);
  }

  // remove + compact: middle
  {
    for (int i = first; i < last; ++i) {
      if (i % 2) {
        std::stringstream ss;
        ss << "test " << i;
        int* entry = map.remove(ss.str());
        ENSURE_EQ(*entry, i);
        delete entry;
      }
    }
    first = (first + 3) / 2;
    for (int i = first, j = 0; j < map.size(); i += 2, ++j) {
      std::stringstream ss;
      ss << "test " << i;
      ENSURE_EQ(*map.get(j).first, ss.str());
      ENSURE_EQ(*map.get(j).second, i);
    }
    map.compact();
    for (int i = first; i < last; i += 2) {
      if (i % 2) {
        std::stringstream ss;
        ss << "test " << i;
        ENSURE_EQ(*map.get(i).first, ss.str());
        ENSURE_EQ(*map.get(i).second, i);
      }
    }

    map.clear();
    ENSURE_EQ(map.size(), 0);
  }

#undef TEST_ORDERED_MAP_SIZE
}

void testVSA801() {
  {
    std::ofstream ofs(TMPFILE, std::ios_base::out);
    ofs << "{\n"
           " \"other_stuff\" : \"no one cares about\", "
           " \"merger\" : {\n"
           "   \"type\" : \"gradient\",\n"
           "   \"blend_radius\" : -1\n"
           " }\n"
           "}\n";
  }

  Parse::JsonDriver driver(false, false);
  ENSURE(driver.parse(TMPFILE));

  ENSURE(driver.getRoot().has("merger"));
  std::unique_ptr<Ptv::Value> mergerConfig(VideoStitch::Ptv::Value::emptyObject());
  mergerConfig->push("merger", driver.getRoot().has("merger")->clone());

  // Brute force method:
  std::stringstream text1, text2;
  driver.getRoot().has("merger")->printJson(text1);
  mergerConfig->has("merger")->printJson(text2);
  ENSURE_EQ(text1.str(), text2.str());

  // Regular method:
  ENSURE(*driver.getRoot().has("merger") == *mergerConfig->has("merger"));
}

void testJsonUtf8() {
  {
    std::string utf8Encoded;
    Util::unicodeToUtf8(0x00e0, utf8Encoded);
    ENSURE_EQ(std::string("\xc3\xa0"), utf8Encoded);
  }
  {
    std::string utf8Encoded;
    Util::unicodeToUtf8(0x00e9, utf8Encoded);
    ENSURE_EQ(std::string("\xc3\xa9"), utf8Encoded);
  }

  Parse::JsonDriver driver(false, false);
  ENSURE(driver.parseData("{\"str\" : \"test \\u00e0v\\u00e9c utf8\"}"));
  ENSURE_EQ(std::string("test \xc3\xa0v\xc3\xa9"
                        "c utf8"),
            driver.getRoot().has("str")->asString());
}

void testVSA1495() {
  {
    std::string utf8Encoded;
    Util::unicodeToUtf8(0x20ac, utf8Encoded);
    ENSURE_EQ(std::string("\xe2\x82\xac"), utf8Encoded);
  }
  Parse::JsonDriver driver(false, false);
  ENSURE(driver.parseData("{\"str\" : \"titi\\u00f9^$\\u20ac~&\"}"));
  ENSURE_EQ(std::string("titi\xc3\xb9^$\xe2\x82\xac~&"), driver.getRoot().has("str")->asString());
}

void testUBJsonCommon(const Ptv::Value& value) {
  ENSURE(value.has("post"));

  ENSURE(value.has("post")->has("id1"));
  ENSURE_EQ(Ptv::Value::INT, value.has("post")->has("id1")->getType());
  ENSURE_EQ((int64_t)-42ll, value.has("post")->has("id1")->asInt());

  ENSURE(value.has("post")->has("id2"));
  ENSURE_EQ(Ptv::Value::INT, value.has("post")->has("id2")->getType());
  ENSURE_EQ((int64_t)-212ll, value.has("post")->has("id2")->asInt());

  ENSURE(value.has("post")->has("id3"));
  ENSURE_EQ(Ptv::Value::INT, value.has("post")->has("id3")->getType());
  ENSURE_EQ((int64_t)-87152ll, value.has("post")->has("id3")->asInt());

  ENSURE(value.has("post")->has("id4"));
  ENSURE_EQ(Ptv::Value::INT, value.has("post")->has("id4")->getType());
  ENSURE_EQ((int64_t)-47865125ll, value.has("post")->has("id4")->asInt());

  ENSURE(value.has("post")->has("id5"));
  ENSURE_EQ(Ptv::Value::INT, value.has("post")->has("id5")->getType());
  ENSURE_EQ((int64_t)0, value.has("post")->has("id5")->asInt());

  ENSURE(value.has("post")->has("d"));
  ENSURE_EQ(Ptv::Value::DOUBLE, value.has("post")->has("d")->getType());
  ENSURE_EQ(-12789654.123, value.has("post")->has("d")->asDouble());
}

void testUBJson() {
  std::cout << "testUBJson" << std::endl;
  Parse::JsonDriver driver(false, false);
  ENSURE(
      driver.parseData("{"
                       "  \"post\": {"
                       "      \"id1\": -42,"
                       "      \"id2\": -212,"
                       "      \"id3\": -87152,"
                       "      \"id4\": -47865125,"
                       "      \"id5\": 0,"
                       "      \"d\": -12789654.123,"
                       "      \"l\": [1, [[]], 1.2],"
                       "      \"author\": \"moi\","
                       "      \"embed\": { \"a\": {} },"
                       "      \"body\": \"titi\xc3\xb9^$\xe2\x82\xac~&\""
                       "  }"
                       "}"));
  testUBJsonCommon(driver.getRoot());

  {
    std::ofstream ofs("ubjson.ubj");
    driver.getRoot().printUBJson(ofs);
  }

  ENSURE(driver.parse("ubjson.ubj"));
  testUBJsonCommon(driver.getRoot());

  std::stringstream ss;
  driver.getRoot().printUBJson(ss);
  ENSURE(driver.parseData(ss.str()));
  testUBJsonCommon(driver.getRoot());
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testPopulate();
  VideoStitch::Testing::testNoFile();
  VideoStitch::Testing::testParse1();
  VideoStitch::Testing::testEscapeFail();
  VideoStitch::Testing::testParseWithPano();
  VideoStitch::Testing::testDefaults("{ \"a\": { \"b\": 2 } }", "{ \"a\": { \"b\": 1 } }", "{ \"a\": { \"b\": 2 } }");
  VideoStitch::Testing::testDefaults("{ \"a\": {          } }", "{ \"a\": { \"b\": 1 } }", "{ \"a\": { \"b\": 1 } }");
  VideoStitch::Testing::testDefaults("{                     }", "{ \"a\": { \"b\": 1 } }", "{                     }");
  VideoStitch::Testing::testDefaults("{                     }", "{ \"l\": [{ \"c\": 3 }] }", "{  \"l\": []          }");
  VideoStitch::Testing::testDefaults("{ \"l\": [{},{}]      }", "{ \"l\": [{ \"c\": 3 }] }",
                                     "{ \"l\": [ { \"c\": 3 }, { \"c\": 3 } ] }");
  VideoStitch::Testing::testDefaults("{ \"l\": [{},{}]      }", "{ \"l\": [{ \"c\": 3 }, { \"d\": 2 }] }",
                                     "{ \"l\": [ { \"c\": 3 }, { \"d\": 2 } ] }");
  VideoStitch::Testing::testDefaults("{ \"l\": [{},{}]      }", "{ \"l\": [] }", "{ \"l\": [{},{}]      }");
  VideoStitch::Testing::testOrderedMap();
  VideoStitch::Testing::testVSA801();
  VideoStitch::Testing::testJsonUtf8();
  VideoStitch::Testing::testVSA1495();
  VideoStitch::Testing::testUBJson();
  return 0;
}
