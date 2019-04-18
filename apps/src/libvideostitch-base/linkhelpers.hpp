// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QString>

static QString LINK_TEMPLATE("<a href=\"%0\" style=\"color:#4282D3; text-decoration:none;\">%1</a>");

/**
 * @brief Creates a formated url using HTML tags.
 * @param url The url of the hiperlink.
 * @param text The text for the URL.
 * @return The formated HTML.
 */
static QString formatLink(const QString url, const QString text = QString()) {
  return LINK_TEMPLATE.arg(url, text.isEmpty() ? url : text);
}
