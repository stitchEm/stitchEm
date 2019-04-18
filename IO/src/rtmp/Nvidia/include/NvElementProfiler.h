/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __NV_ELEMENT_PROFILER_H__
#define __NV_ELEMENT_PROFILER_H__

#include <iostream>
#include <pthread.h>
#include <map>
#include <stdint.h>
#include <sys/time.h>

/**
 *
 * @c %NvElementProfiler is a helper class for profiling the performance of individual
 * elements.
 *
 * NvElementProfiler currently measures processing latencies, average processing rate and
 * number of units which arrived late at the element. This should be used internally
 * by the components.
 *
 * Components should call startProcessing() to indicate that a unit has been submitted
 * for processing and finishProcessing() to indicate that a unit has finished processing.
 * Components who do not require latency measurement need not call startProcessing().
 *
 * Components can get data from NvElementProfiler using getProfilerData(). It
 * fills the #NvElementProfilerData structure. Components might not support all
 * the fields available in the strcuture and so a variable #valid_fields of
 * type #ProfilerField is also included in the structure.
 */
class NvElementProfiler {
 public:
  /**
   * Data type indicating valid fields in #NvElementProfilerData structure.
   */
  typedef int ProfilerField;
  static const ProfilerField PROFILER_FIELD_NONE = 0;
  static const ProfilerField PROFILER_FIELD_TOTAL_UNITS = 1;
  static const ProfilerField PROFILER_FIELD_LATE_UNITS = 2;
  static const ProfilerField PROFILER_FIELD_LATENCIES = 4;
  static const ProfilerField PROFILER_FIELD_FPS = 8;
  static const ProfilerField PROFILER_FIELD_ALL = (PROFILER_FIELD_FPS << 1) - 1;

  /**
   * Holds profiling data for the element.
   *
   * Some elements may not support all the fields in the structure. User should check
   * the valid_fields flag to check which fields are valid.
   */
  typedef struct {
    /** Valid Fields which are supported by the element. */
    ProfilerField valid_fields;

    /** Average latency of all processed units, in microseconds. */
    uint64_t average_latency_usec;
    /** Minimum of latencies for each processed units, in microseconds. */
    uint64_t min_latency_usec;
    /** Maximum of latencies for each processed units, in microseconds. */
    uint64_t max_latency_usec;

    /** Total units processed. */
    uint64_t total_processed_units;
    /** Number of units which arrived late at the element. */
    uint64_t num_late_units;

    /** Average rate at which the units were processed. */
    float average_fps;

    /** Total profiling time. */
    struct timeval profiling_time;
  } NvElementProfilerData;

  /**
   * Get the profiling data for the element.
   *
   * @param[out] data Reference to the NvElementProfilerData structure which should be filled.
   */
  void getProfilerData(NvElementProfilerData &data);

  /**
   * Print the element's profiling data to an output stream.
   *
   * @param[in] out_stream Reference to a std::ostream.
   */
  void printProfilerData(std::ostream &out_stream = std::cout);

  /**
   * Inform the profiler processing has started.
   *
   * Has no effect if profiler is disabled.
   *
   * @return ID of the unit, to be supplied with finishProcessing();.
   */
  uint64_t startProcessing();

  /**
   * Inform the profiler processing has finished.
   *
   * Has no effect if profiler is disabled.
   *
   * @param[in] id ID of the unit whose processing is finished,
   *          0 if the first unit in the profiler's queue should be picked.
   * @param[in] is_late Should be true if the frame arrived late at the element.
   */
  void finishProcessing(uint64_t id, bool is_late);

  /**
   * Enable the profiler.
   *
   * startProcessing() and finishProcessing() will not have any effect till the profiler is enabled.
   *
   * @param[in] reset_data Reset the profiled data.
   */
  void enableProfiling(bool reset_data);

  /**
   * Disable the profiler.
   */
  void disableProfiling();

 private:
  /**
   * Reset the profiler data.
   */
  void reset();

  pthread_mutex_t profiler_lock; /**< Mutex to synchronize multithreaded access to profiler data. */

  bool enabled; /**< Flag indicating if profiler is enabled. */

  const ProfilerField valid_fields; /**< Valid fields for the element. */

  struct NvElementProfilerDataInternal : NvElementProfilerData {
    /** Wall-clock time at which the first unit was processed. */
    struct timeval start_time;

    /** Wall-clock time at which the latest unit was processed. */
    struct timeval stop_time;

    /** Total accumulated time.
     *  When performance measurement is restarted #start_time and #stop_time
     *  will be reset. This field is used to accumulate time before
     *  resetting. */
    struct timeval accumulated_time;

    /** Total accumulated latency for all units, in microseconds. */
    uint64_t total_latency;
  } data_int;

  /** Queue used to maintain the timestamps of when the unit
   *  processing started. Required to calculate latency. */
  std::map<uint64_t, struct timeval> unit_start_time_queue;

  uint64_t unit_id_counter; /**< Unique ID of the last unit. */

  /**
   * Constructor for NvElementProfiler.
   *
   * Initializes internal data structures. The profiler is disabled by default.
   * @param fields
   */
  NvElementProfiler(ProfilerField fields);

  /**
   * Disallow copy constructor.
   */
  NvElementProfiler(const NvElementProfiler &that);
  /**
   * Disallow assignment.
   */
  void operator=(NvElementProfiler const &);

  ~NvElementProfiler();

  friend class NvElement;
};

#endif
