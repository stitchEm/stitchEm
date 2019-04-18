// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef VSLOG_HPP
#define VSLOG_HPP

#include "common-config.hpp"

#include <QObject>

#include <vector>
#include <iostream>

namespace VideoStitch {
namespace Helper {

class VS_COMMON_EXPORT VSLog : public QObject, public std::streambuf {
  Q_OBJECT

 public:
  explicit VSLog(std::size_t buff_sz = 1024);
  virtual ~VSLog();
  static QString camelCaseToSpace(const QString& camelCasedString);
 signals:
  void emitError(QString);

 private:
  VSLog(const VSLog&);
  VSLog& operator=(const VSLog&);

  int sync() override;

  std::vector<char> buffer_;
};

}  // namespace Helper
}  // namespace VideoStitch
#endif
