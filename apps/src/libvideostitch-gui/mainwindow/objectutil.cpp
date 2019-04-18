// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "objectutil.hpp"

namespace VideoStitch {
namespace Helper {
void toggleConnect(bool connectionState, QObject *sender, const char *signal, QObject *receiver, const char *slot,
                   Qt::ConnectionType flag) {
  if (connectionState) {
    QObject::connect(sender, signal, receiver, slot, flag);
  } else {
    QObject::disconnect(sender, signal, receiver, slot);
  }
}
}  // namespace Helper
}  // namespace VideoStitch
