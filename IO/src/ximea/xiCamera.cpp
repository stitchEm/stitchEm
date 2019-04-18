// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifdef linux
// linux
#else
// windows
#include <Windows.h>
#endif
#include <xiCamera.h>
#include <memory.h>

// ----------------------------------------------------
// constructor - initialize class variables

xi4Camera::xi4Camera() {
  handle = NULL;
  memset(hBuffer, 0, sizeof(hBuffer));
  memset(Buffer_Dat, 0, sizeof(Buffer_Dat));
  memset(Buffer_Private, 0, sizeof(Buffer_Private));
  NewBufferEvent = NULL;
  memset(&NewBufferInfo, 0, sizeof(NewBufferInfo));
  buffers_allocated = false;
}

// ----------------------------------------------------
// destructor - remove all buffers

xi4Camera::~xi4Camera() { BuffersDeAllocate(); }

// ----------------------------------------------------
void xi4Camera::CloseDevice() {
  BuffersDeAllocate();
  fhandler(xiCloseDevice(handle));
}

// ----------------------------------------------------
// allocate camera buffers

void xi4Camera::BuffersAllocate() {
  for (int i = 0; i < XICAMERA_BUFF_COUNT; i++) {
    // Allocate memory
#ifdef linux
    posix_memalign(&Buffer_Dat[i], DAL_PCIE_PLDA_PAGE_SIZE_BYTES, XICAMERA_BUFF_SIZE);
#else
    Buffer_Dat[i] = _aligned_malloc(XICAMERA_BUFF_SIZE, DAL_PCIE_PLDA_PAGE_SIZE_BYTES);
#endif
    memset(Buffer_Dat[i], 0, XICAMERA_BUFF_SIZE);
    // Set Pointer to Private data (could be used as a pointer to user defined structure or data)
    Buffer_Private[i] = Buffer_Dat[i];
    // Announce Buffer
    AnnounceBuffer(Buffer_Dat[i], XICAMERA_BUFF_SIZE, Buffer_Private[i], &hBuffer[i]);
  }
  // register new buffer event
  UnRegisterEvent(XI_EVENT_NEW_BUFFER);
  RegisterEvent(XI_EVENT_NEW_BUFFER, &NewBufferEvent);
  buffers_allocated = true;
}

// ----------------------------------------------------
// allocate camera buffers

void xi4Camera::BuffersDeAllocate() {
  if (!buffers_allocated) return;

  for (int i = 0; i < XICAMERA_BUFF_COUNT; i++) {
    // De-Allocate memory
    void* pBuffer[XICAMERA_BUFF_COUNT];
    void* pPrivate[XICAMERA_BUFF_COUNT];

    RevokeBuffer(hBuffer[i], &pBuffer[i], &pPrivate[i]);
#ifdef linux
    free(pBuffer[i]);
#else
    _aligned_free((LPVOID)pBuffer[i]);

#endif
  }
  buffers_allocated = false;
}

// ----------------------------------------------------
// Queue Buffer

void xi4Camera::BuffersQueue(int queue_count) {
  if (queue_count > XICAMERA_BUFF_COUNT) throw "xi4Camera::BuffersQueue invalid queue_count";
  for (int j = 0; j < queue_count; j++) {
    QueueBuffer(hBuffer[j]);
  }
}

// ----------------------------------------------------
// wait for next image

xi_return_e xi4Camera::WaitForNextImage(int timeout_ms) {
  UINT32 BufferSize = sizeof(NewBufferInfo);

  return EventGetData(NewBufferEvent, &NewBufferInfo, &BufferSize, timeout_ms);
}
