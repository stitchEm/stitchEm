// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QString>

class Codec;

/**
 * @brief The CodecFactory class is a class used to create a codec given a key which identifies this codec.
 */
class VS_GUI_EXPORT CodecFactory {
 public:
  static Codec* create(const QString& key, QWidget* parent);
};
