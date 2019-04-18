// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef OBJECTUTIL_HPP
#define OBJECTUTIL_HPP

#include <QObject>
namespace VideoStitch {
namespace Helper {
/**
 * @brief Method which allows to connect/disconnect using a boolean.
 * @param True = connect the objects together, false = disconnect them
 * @param Sender object
 * @param Signal to connect from
 * @param Receiver object
 * @param Slot to connect to (that can be another signal)
 * @param Qt connection flag, the default flag is AutoConnection, as it is when you use a classic QObejct::connect
 */
void VS_GUI_EXPORT toggleConnect(bool connectionState, QObject *sender, const char *signal, QObject *receiver,
                                 const char *slot, Qt::ConnectionType flag = Qt::AutoConnection);
}  // namespace Helper
}  // namespace VideoStitch

#endif  // OBJECTUTIL_HPP
