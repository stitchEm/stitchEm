// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ximeaReader.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

#include <cstdint>
#include <sstream>

// GPU//#include <cuda_runtime.h>

using namespace VideoStitch;
using namespace Input;

DualBuffer::DualBuffer(int64_t widthParam, int64_t heightParam)
    : readPtr(0), writePtr(0), size(widthParam * heightParam), empty(true), full(false), stopLoop(false) {
  for (uint8_t i = 0; i < 2; i++) {
    buff[i].second = new unsigned char[size];
    std::memset(buff[i].second, 0, size * sizeof(unsigned char));
  }

  wCond.notify_one();
  wCond.notify_one();
}

DualBuffer::~DualBuffer() {
  for (uint8_t i = 0; i < 2; i++) {
    delete buff[i].second;
  }
}

bool DualBuffer::read(XFrame* data) {
  std::unique_lock<std::mutex> lock(mtx);
  rCond.wait(lock, [this]() { return !empty || stopLoop; });

  if (stopLoop) {
    return false;
  }

  data->first = buff[readPtr].first;
  std::memcpy(data->second, buff[readPtr].second, size);
  if (readPtr == 0) {
    readPtr = 1;
  } else
    readPtr = 0;
  full = false;

  if (readPtr == writePtr) {
    empty = true;
  }

  wCond.notify_one();

  return true;
}

bool DualBuffer::write(XFrame* data) {
  std::unique_lock<std::mutex> lock(mtx);
  wCond.wait(lock, [this]() { return !full || stopLoop; });

  if (stopLoop) {
    return false;
  }

  buff[writePtr].first = data->first;
  std::memcpy(buff[writePtr].second, data->second, size);
  if (writePtr == 0) {
    writePtr = 1;
  } else
    writePtr = 0;
  empty = false;

  if (readPtr == writePtr) {
    full = true;
  }

  rCond.notify_one();

  return true;
}

XimeaReader::XimeaReader(readerid_t id, const int64_t width, const int64_t height, int deviceIndex,
                         const bool withAudio, FrameRate fps, bool interlaced, int bw, int frameRateLimit)
    : Reader(id),
      VideoReader(width, height, width * height, Grayscale, Host, fps,
                  // GPU//3 * width * height, RGB, Device, fps,
                  0, NO_LAST_FRAME, false /* not a procedural reader */, nullptr),
      devIdx(deviceIndex),
      bandwidth(bw),
      fpsLimit(frameRateLimit),
      stop(false),
      dBuff(new DualBuffer(width, height)) {
  thr = new std::thread(&XimeaReader::ximeaThread, this);
}

XimeaReader::~XimeaReader() {
  stop = true;
  dBuff->stop();
  thr->join();
  delete thr;
  thr = nullptr;

  delete dBuff;
  dBuff = nullptr;
}

XimeaReader* XimeaReader::create(readerid_t id, const Ptv::Value* config, const int64_t width, const int64_t height) {
  // const UWord deviceIndex,
  const bool withAudio = false;
  bool interlaced = false;

  int deviceIndex = 0;
  if (Parse::populateInt("Ximea reader", *config, "device", deviceIndex, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "[Ximea] Cam device index (\"device\") couldn't be retrieved. Aborting." << std::endl;
    return nullptr;
  }

  FrameRate frameRate;
  if (!config->has("frame_rate")) {
    Logger::get(Logger::Error) << "Frame rate (\"frame_rate\") couldn't be retrieved. Aborting." << std::endl;
    return nullptr;
  } else {
    const Ptv::Value* fpsConf = config->has("frame_rate");
    if ((Parse::populateInt("Ximea reader", *fpsConf, "num", frameRate.num, false) !=
         VideoStitch::Parse::PopulateResult_Ok) ||
        (Parse::populateInt("Ximea reader", *fpsConf, "den", frameRate.den, false) !=
         VideoStitch::Parse::PopulateResult_Ok)) {
      Logger::get(Logger::Error) << "[Ximea] Frame rate (\"frame_rate\") couldn't be retrieved. Aborting." << std::endl;
      return nullptr;
    }
  }

  int bw = 0;
  if (Parse::populateInt("Ximea reader", *config, "bandwidth", bw, false) != VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Warning)
        << "[Ximea] Cam bandwidth limit (\"bandwidth\") couldn't be retrieved. Automatic bandwidth calculus used."
        << std::endl;
  }

  int fpsLim = 0;
  if (Parse::populateInt("Ximea reader", *config, "fps_limit", fpsLim, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Warning)
        << "[Ximea] Cam framerate limit (\"fps_limit\") couldn't be retrieved. Automatic framerate calculus used."
        << std::endl;
  }

  return new XimeaReader(id, width, height, deviceIndex, withAudio, frameRate, interlaced, bw, fpsLim);
  ;
}

ReadStatus XimeaReader::readFrame(mtime_t& date, unsigned char* video) {
  XFrame tmp;
  std::memset(video, 0, getWidth() * getHeight());
  // GPU//cudaMemset(video, 0, getWidth()*getHeight()*3);

  date = tmp.first;
  tmp.second = video;
  if (!dBuff->read(&tmp)) {
    return ReadStatus::fromCode<ReadStatusCode::EndOfFile>();
  }

  /*GPU
  if(!imgQueue.empty()){
                unsigned char* frame = imgQueue.front();
                imgQueue.pop();
                cudaMemcpy(video, frame, getWidth()*getHeight(), cudaMemcpyDeviceToDevice );
                cudaMemcpy(video, frame, getWidth()*getHeight(), cudaMemcpyDeviceToDevice );
                cudaMemcpy(video, frame, getWidth()*getHeight(), cudaMemcpyDeviceToDevice );

  }*/

  return ReadStatus::OK();
}

Status XimeaReader::seekFrame(frameid_t) { return Status::OK(); }

void XimeaReader::ximeaThread(XimeaReader* XR) {
  HANDLE xHdl;
  XI_IMG xImg;
  int param = 0;

  memset(&xImg, 0, sizeof(xImg));
  xImg.size = sizeof(XI_IMG);
  // GPU//xImg.frm = XI_FRM_TRANSPORT_DATA;
  xImg.bp_size = (DWORD)(XR->getWidth() * XR->getHeight());

  XI_RETURN stat = xiOpenDevice(XR->deviceIndex(), &xHdl);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error opening cam : " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }

  // stupid flanders, black image otherwise
  stat = xiSetParamInt(xHdl, XI_PRM_EXPOSURE, 20000);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error setting exposition : " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }

  std::stringstream msg;
  msg << "[Ximea] cam  number : " << XR->deviceIndex();  // << std::endl;

  /*
   * GPU
   *//*
stat = xiSetParamInt(xHdl,  XI_PRM_TRANSPORT_DATA_TARGET, XI_TRANSPORT_DATA_TARGET_GPU_RAM);
if(stat != XI_OK) {
	Logger::get(Logger::Error) << "[Ximea] Error setting GPU memory : " << XR->deviceIndex() << " error : " << stat << std::endl;
}

stat = xiSetParamInt(xHdl, XI_PRM_IMAGE_DATA_FORMAT, XI_FRM_TRANSPORT_DATA);
if(stat != XI_OK) {
	Logger::get(Logger::Error) << "[Ximea] Error setting data format : " << XR->deviceIndex() << " error : " << stat << std::endl;
}
*/

  /*
   * FRAMERATE
   */
  if (XR->fpsLim() != 0) {
    stat = xiSetParamInt(xHdl, XI_PRM_ACQ_TIMING_MODE, XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT);
    if (stat != XI_OK) {
      Logger::get(Logger::Error) << "[Ximea] Error setting framerate accuracy : " << XR->deviceIndex()
                                 << " error : " << stat << std::endl;
    }

    param = (int)XR->fpsLim();
    stat = xiSetParamInt(xHdl, XI_PRM_FRAMERATE, param);
    if (stat != XI_OK) {
      Logger::get(Logger::Error) << "[Ximea] Error setting framerate  : " << XR->deviceIndex() << " error : " << stat
                                 << std::endl;
    }
    msg << " Framerate set : " << param;
  } else {
    Logger::get(Logger::Info) << "[Ximea] Using automatic framerate calculation" << std::endl;
  }

  stat = xiGetParamInt(xHdl, XI_PRM_FRAMERATE, &param);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error getting framerate " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }
  msg << " Framerate get : " << param << std::endl;

  /*
   * BANDWIDTH
   */
  int bwMax = 0;
  int bwMin = 0;

  stat = xiGetParamInt(xHdl, XI_PRM_AUTO_BANDWIDTH_CALCULATION, &param);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error getting bandwidth calc : " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }
  msg << " Auto bandwidth calculation : " << param;

  stat = xiGetParamInt(xHdl, XI_PRM_LIMIT_BANDWIDTH_MODE, &param);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error getting bandwidth mode : " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }
  msg << " Bandwidth limit mode : " << param;

  stat = xiGetParamInt(xHdl, XI_PRM_LIMIT_BANDWIDTH, &param);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error getting bandwidth : " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }
  msg << " Bandwidth limit  : " << param << std::endl;

  stat = xiGetParamInt(xHdl, XI_PRM_LIMIT_BANDWIDTH XI_PRM_INFO_MIN, &bwMin);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error getting bandwidth min : " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }
  msg << " Bandwidth min : " << bwMin;

  stat = xiGetParamInt(xHdl, XI_PRM_LIMIT_BANDWIDTH XI_PRM_INFO_MAX, &bwMax);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error getting bandwidth max : " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }
  msg << " Bandwidth max : " << bwMax << std::endl;

  Logger::get(Logger::Error) << msg.str();

  if (XR->bw() != 0) {
    if ((XR->bw() > bwMin) && (XR->bw() < bwMax)) {
      stat = xiSetParamInt(xHdl, XI_PRM_AUTO_BANDWIDTH_CALCULATION, XI_OFF);
      if (stat != XI_OK) {
        Logger::get(Logger::Error) << "[Ximea] Error setting bandwidth calc : " << XR->deviceIndex()
                                   << " error : " << stat << std::endl;
      }

      stat = xiSetParamInt(xHdl, XI_PRM_LIMIT_BANDWIDTH_MODE, XI_ON);
      if (stat != XI_OK) {
        Logger::get(Logger::Error) << "[Ximea] Error setting bandwidth mode : " << XR->deviceIndex()
                                   << " error : " << stat << std::endl;
      }

      stat = xiSetParamInt(xHdl, XI_PRM_LIMIT_BANDWIDTH, (int)XR->bw());
      if (stat != XI_OK) {
        Logger::get(Logger::Error) << "[Ximea] Error setting bandwidth : " << XR->deviceIndex() << " error : " << stat
                                   << std::endl;
      }
      msg << " bandwidth limit set : " << XR->bw();
    } else {
      Logger::get(Logger::Error) << "[Ximea] Error settings bandwidth, user value outside bond : " << XR->bw()
                                 << " is not between " << bwMin << " and " << bwMax << std::endl;
    }
  } else {
    Logger::get(Logger::Warning) << "[Ximea] Using automatic bandwidth calculation" << std::endl;
  }

  stat = xiGetParamInt(xHdl, XI_PRM_AUTO_BANDWIDTH_CALCULATION, &param);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error getting bandwidth calc : " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }
  msg << " Auto bandwidth calculation : " << param;

  stat = xiGetParamInt(xHdl, XI_PRM_LIMIT_BANDWIDTH_MODE, &param);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error getting bandwidth mode : " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }
  msg << " Bandwidth limit mode : " << param;

  stat = xiGetParamInt(xHdl, XI_PRM_LIMIT_BANDWIDTH, &param);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error getting bandwidth : " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }
  msg << " Bandwidth limit  : " << param << std::endl;

  /*
   * Buffer policy
   *//*
stat = xiSetParamInt(xHdl, XI_PRM_BUFFER_POLICY, XI_BP_SAFE);
if(stat != XI_OK) {
	Logger::get(Logger::Error) << "[Ximea] Error settin buffer policy : " << XR->deviceIndex() << " error : " << stat << std::endl;
}


stat = xiGetParamInt(xHdl, XI_PRM_BUFFER_POLICY, &param);
if(stat != XI_OK) {
	Logger::get(Logger::Error) << "[Ximea] Error getting buffer policy : " << XR->deviceIndex() << " error : " << stat << std::endl;
}
msg << " buffer policy : " << param << std::endl;
*/

  /*
   * Cam sync
   *//*
  if(XR->deviceIndex() == 0){
    stat = xiSetParamInt(xHdl, XI_PRM_GPO_SELECTOR, XI_GPO_PORT1);
    if(stat != XI_OK) {
      Logger::get(Logger::Error) << "[Ximea] Error setting GPO for main cam : " << XR->deviceIndex() << " error : " << stat << std::endl;
    }

    stat = xiSetParamInt(xHdl, XI_PRM_GPO_MODE, XI_GPO_FRAME_ACTIVE);
    if(stat != XI_OK) {
      Logger::get(Logger::Error) << "[Ximea] Error set GPO mode for main cam : " << XR->deviceIndex() << " error : " << stat << std::endl;
    }
  }else{
    stat = xiSetParamInt(xHdl, XI_PRM_GPI_SELECTOR, XI_GPI_PORT1);
    if(stat != XI_OK) {
      Logger::get(Logger::Error) << "[Ximea] Error setting GPI for slave cam : " << XR->deviceIndex() << " error : " << stat << std::endl;
    }

    stat = xiSetParamInt(xHdl, XI_PRM_GPI_MODE, XI_GPI_TRIGGER);
    if(stat != XI_OK) {
      Logger::get(Logger::Error) << "[Ximea] Error set GPI mode for slave cam : " << XR->deviceIndex() << " error : " << stat << std::endl;
    }

    stat = xiSetParamInt(xHdl, XI_PRM_TRG_SOURCE, XI_TRG_EDGE_RISING);
    if(stat != XI_OK) {
      Logger::get(Logger::Error) << "[Ximea] Error set trigger source for slave cam : " << XR->deviceIndex() << " error : " << stat << std::endl;
    }

    stat = xiSetParamInt(xHdl, XI_PRM_TRG_SELECTOR, XI_TRG_SEL_FRAME_START);
    if(stat != XI_OK) {
      Logger::get(Logger::Error) << "[Ximea] Error set trigger selector for slave cam : " << XR->deviceIndex() << " error : " << stat << std::endl;
    }

  }*/

  Logger::get(Logger::Error) << msg.str();

  std::this_thread::sleep_for(std::chrono::milliseconds(10));  // allow other camera to start

  stat = xiStartAcquisition(xHdl);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error starting acquisition : " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }

  /*
   * MAIN LOOP
   */
  XFrame tmpFrame;
  while (!XR->stopThread()) {
    /*GPU
    unsigned char* frameBuffer;
      cudaMalloc((void**)&frameBuffer, getWidth()*getHeight());
    */

    stat = xiGetImage(xHdl, 500, &xImg);
    if (stat != XI_OK) {
      Logger::get(Logger::Error) << "[Ximea] Error getting image from cam : " << XR->deviceIndex()
                                 << " error : " << stat << std::endl;
    } else {
      tmpFrame.first = (mtime_t)xImg.tsSec * 1000000 + (mtime_t)xImg.tsUSec;
      tmpFrame.second = (unsigned char*)xImg.bp;

      XR->dBuff->write(&tmpFrame);

      // cudaMemcpy(frameBuffer, xImg.bp, getWidth()*getHeight(), cudaMemcpyDeviceToDevice );
      // GPU//XR->imgQueue.push(frameBuffer);
    }
  }

  stat = xiStopAcquisition(xHdl);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error stopping acquisition : " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }

  stat = xiCloseDevice(xHdl);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error closing cam : " << XR->deviceIndex() << " error : " << stat
                               << std::endl;
  }
}
