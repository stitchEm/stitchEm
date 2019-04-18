// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rtmpDiscovery.hpp"
#include "videoEncoder.hpp"

namespace VideoStitch {
namespace Plugin {

RTMPDiscovery* RTMPDiscovery::create() { return new RTMPDiscovery(); }

RTMPDiscovery::RTMPDiscovery() : m_supportedCodecs() { Output::VideoEncoder::supportedCodecs(m_supportedCodecs); }

RTMPDiscovery::~RTMPDiscovery() {}

std::vector<std::string> RTMPDiscovery::supportedVideoCodecs() { return m_supportedCodecs; }

std::string RTMPDiscovery::name() const { return "rtmp"; }

std::string RTMPDiscovery::readableName() const { return "RTMP"; }

std::vector<Plugin::DiscoveryDevice> RTMPDiscovery::outputDevices() { return std::vector<Plugin::DiscoveryDevice>(); }

std::vector<Plugin::DiscoveryDevice> RTMPDiscovery::inputDevices() { return std::vector<Plugin::DiscoveryDevice>(); }
}  // namespace Plugin
}  // namespace VideoStitch
