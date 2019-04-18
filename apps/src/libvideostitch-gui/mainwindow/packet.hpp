// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QVariant>
#include <QDataStream>

/**
 * @brief Class used to encapsulate a packet which may be sent to another version of VS.
 */
class VS_GUI_EXPORT Packet {
 public:
  enum Types { WAKEUP = 0x01, OPEN_FILES = 0x02, UNKNOWN = 0x00 };

  Packet(const quint8 type = UNKNOWN, const quint16 payloadSize = 0, const QByteArray payload = QByteArray());
  ~Packet() {}
  Packet(const Packet& copie);
  /**
   * @brief Returns the type of the packet
   */
  quint8 getType() const;
  /**
   * @brief Returns the payload size of the packet
   */
  quint16 getPayloadSize() const;
  /**
   * @brief Get the payload of the packet
   */
  QByteArray getPayload() const;

 private:
  quint8 type;
  quint16 payloadSize;
  QByteArray payload;
  /**
   * @brief To send a packet through a data stream
   * @param out input data stream
   * @param value packet to send
   * @return filled data stream
   */
  friend QDataStream& operator<<(QDataStream& out,
                                 const Packet& value); /**
                                                        * @brief constructs a packed from a data stream
                                                        * @param in input stream
                                                        * @param value Resulting packet
                                                        * @return emptied stream
                                                        */
  friend QDataStream& operator>>(QDataStream& in, Packet& value);
};

Q_DECLARE_METATYPE(Packet)
QDataStream& operator<<(QDataStream& out, const Packet& value);
QDataStream& operator>>(QDataStream& in, Packet& value);
