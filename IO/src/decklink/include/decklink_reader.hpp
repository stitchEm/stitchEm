// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/profile.hpp"
#include "libvideostitch/input.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/plugin.hpp"

#if defined(_WIN32)
#include "DeckLinkAPI_h.h"
#else
#include "DeckLinkAPI.h"
#endif

#include <condition_variable>
#include <mutex>
#include <memory>

/**
 * BlackMagic DeckLink capture card reader.
 */
namespace VideoStitch {
namespace Input {

class DeckLinkReader : public VideoReader, public IDeckLinkInputCallback {
 public:
  virtual ~DeckLinkReader();

  static DeckLinkReader* create(readerid_t id, const Ptv::Value* config, const int64_t width, const int64_t height);
  static bool handles(const Ptv::Value* config);

  ReadStatus readFrame(mtime_t& timestamp, unsigned char* videoFrame);
  Status seekFrame(frameid_t date);

  HRESULT QueryInterface(REFIID riid, void** ppv) { return E_FAIL; }
  ULONG AddRef() { return 0; }
  ULONG Release() { return 0; }

  HRESULT VideoInputFrameArrived(IDeckLinkVideoInputFrame*, IDeckLinkAudioInputPacket*);
  HRESULT VideoInputFormatChanged(BMDVideoInputFormatChangedEvents, IDeckLinkDisplayMode*,
                                  BMDDetectedVideoInputFormatFlags);

 private:
  DeckLinkReader(readerid_t id, int64_t width, int64_t height, PixelFormat pixelFormat, Plugin::DisplayMode displayMode,
                 int64_t frameSize, FrameRate frameRate, std::shared_ptr<IDeckLink> subDevice,
                 std::shared_ptr<IDeckLinkInput> input, std::string name);

  std::shared_ptr<IDeckLink> subDevice;
  std::shared_ptr<IDeckLinkConfiguration> configurationForHalfDuplex;  // We need to keep the configuration because
  // In Decklink SDK doc: "Changes will persist until the IDeckLinkConfiguration object is released"
  // Be careful: the configuration is not necessarily the one of the above sub device. It can be the configuration of
  // the paired sub device (for Quad 2 and Duo 2)
  std::shared_ptr<IDeckLinkInput> input;
  const std::string name;
  const Plugin::DisplayMode displayMode;

  std::mutex m;
  std::condition_variable cv;
  std::vector<unsigned char> currentVideoFrame;
  bool frameAvailable;
  mtime_t videoTimeStamp;
};

}  // namespace Input
}  // namespace VideoStitch
