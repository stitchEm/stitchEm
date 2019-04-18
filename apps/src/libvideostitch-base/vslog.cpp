// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "vslog.hpp"

#include <QStringList>

#include <streambuf>

namespace VideoStitch {
namespace Helper {

VSLog::VSLog(std::size_t buff_sz) : buffer_(buff_sz + 1) {
  char* base = &buffer_.front();
  setp(base, base + buff_sz);
}

VSLog::~VSLog() {}

QString VSLog::camelCaseToSpace(const QString& camelCasedString) {
  QStringList words = camelCasedString.split(QRegExp("(?=[A-Z])"), QString::SkipEmptyParts);
  QString spacedSentence;
  for (int i = 0; i < words.size(); i++) {
    if (i > 0) {
      spacedSentence += " " + words[i];
    } else {
      spacedSentence += words[i].left(1).toUpper() + words[i].mid(1);
    }
  }
  return spacedSentence;
}

int VSLog::sync() {
  int n = int(pptr() - pbase());
  if (n < 0) {
    return -1;
  }
  std::string temp;
  temp.assign(pbase(), n);
  pbump(-n);
  std::cerr << temp.c_str() << std::flush;
  emit emitError(QString::fromUtf8(temp.c_str()));
  return 0;
}
}  // namespace Helper
}  // namespace VideoStitch
