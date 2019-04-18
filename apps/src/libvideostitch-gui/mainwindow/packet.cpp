// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "packet.hpp"

Packet::Packet(const Packet& copy) {
  type = copy.type;
  payloadSize = copy.payloadSize;
  payload = copy.payload;
}

Packet::Packet(const quint8 ty, const quint16 paySize, const QByteArray pay)
    : type(ty), payloadSize(paySize), payload(pay) {}

quint8 Packet::getType() const { return type; }

quint16 Packet::getPayloadSize() const { return payloadSize; }

QByteArray Packet::getPayload() const { return payload; }

QDataStream& operator<<(QDataStream& out, const Packet& value) {
  out << value.type << value.payloadSize << value.payload;
  return out;
}

QDataStream& operator>>(QDataStream& in, Packet& value) {
  in >> value.type >> value.payloadSize >> value.payload;
  return in;
}
