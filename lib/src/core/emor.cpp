// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/emor.hpp"
#include "emor_coefs.hpp"
#include "emor_inv_coefs.hpp"

#include <algorithm>
#include <cassert>

namespace VideoStitch {
namespace Core {

ResponseCurve::ResponseCurve() : responseCurve(new float[totalLutSize()]), monotonyError(-1) {
  for (int i = 0; i < totalLutSize(); ++i) {
    responseCurve[i] = 0.0f;
  }
}

ResponseCurve::~ResponseCurve() { delete[] responseCurve; }

const float* ResponseCurve::getResponseCurve() const { return responseCurve; }

const float* ResponseCurve::getInverseResponseCurve() const { return responseCurve + lutSize(); }

int ResponseCurve::getMonotonyError() const { return monotonyError; }

void ResponseCurve::invert() {
  for (int i = 0; i < lutSize(); ++i) {
    std::swap(*(responseCurve + i), *(responseCurve + i + lutSize()));
  }
}

void ResponseCurve::calculateMonotonyError() {
  monotonyError = 0;
  int lastVal = 0;
  for (int i = 1; i < lutSize() - 1; ++i) {
    if (i % 4 == 0) {
      int val = (int)(responseCurve[i] * 1024);
      if (val <= lastVal) {
        monotonyError++;
      }
      lastVal = std::max(val, lastVal);
    }
  }
}

void ResponseCurve::makeMonotonous() {
  calculateMonotonyError();

  float maxVal = responseCurve[lutSize() - 1];
  for (int i = 0; i < lutSize() - 1; ++i) {
    if (responseCurve[i + 1] > maxVal) {
      responseCurve[i + 1] = maxVal;
    } else if (responseCurve[i + 1] < responseCurve[i]) {
      responseCurve[i + 1] = responseCurve[i];
    }
  }
}

void ResponseCurve::computeInverseResponse() {
  makeMonotonous();
  float* inverseResponseCurve = responseCurve + lutSize();
  for (int i = 0; i < lutSize() - 1; ++i) {
    const float y0 = lutSize() * responseCurve[i];
    const float y1 = lutSize() * responseCurve[i + 1];
    for (int j = (int)y0; j <= (int)y1 && j < lutSize(); ++j) {
      if (y1 == y0) {
        inverseResponseCurve[j] = y0;
      } else {
        const float x = ((float)j - y0) / (y1 - y0);
        inverseResponseCurve[j] = ((float)i + x) / lutSize();
      }
    }
  }
}

ValueResponseCurve::ValueResponseCurve(const std::array<uint16_t, 256>& values) {
#if (!_MSC_VER || _MSC_VER >= 1900)
  static_assert(4 * std::remove_reference<decltype(values)>::type().size() == lutSize(),
                "Expecting to fill curve by interpolating from 256 to 1024 values");
#endif

  for (size_t index10 = 0; index10 < lutSize(); index10++) {
    const float index8f = (float)index10 / 1023.f * 255.f;
    assert(index8f >= 0.f && index8f < (float)values.size());
    const int lowerIndex8i = std::max(0, std::min((int)index8f, (int)values.size() - 1));
    const int upperIndex8i = std::min(lowerIndex8i + 1, (int)values.size() - 1);
    const float x = index8f - (float)lowerIndex8i;
    float val = (1.0f - x) * values[lowerIndex8i] + x * values[upperIndex8i];
    responseCurve[index10] = val / 1023.f;
  }
  computeInverseResponse();
}

EmorResponseCurve::EmorResponseCurve(double emor1, double emor2, double emor3, double emor4, double emor5)
    : ResponseCurve() {
  for (int i = 0; i < lutSize(); ++i) {
    responseCurve[i] = f0[i];
    responseCurve[i] += (float)emor1 * h1[i];
    responseCurve[i] += (float)emor2 * h2[i];
    responseCurve[i] += (float)emor3 * h3[i];
    responseCurve[i] += (float)emor4 * h4[i];
    responseCurve[i] += (float)emor5 * h5[i];
    if (responseCurve[i] < 0.0f) {
      responseCurve[i] = 0.0f;
    } else if (responseCurve[i] > 1.0f) {
      responseCurve[i] = 1.0f;
    }
  }
  computeInverseResponse();
}

InvEmorResponseCurve::InvEmorResponseCurve(double emor1, double emor2, double emor3, double emor4, double emor5)
    : ResponseCurve() {
  for (int i = 0; i < lutSize(); ++i) {
    responseCurve[i] = fInv0[i];
    responseCurve[i] += (float)emor1 * hInv1[i];
    responseCurve[i] += (float)emor2 * hInv2[i];
    responseCurve[i] += (float)emor3 * hInv3[i];
    responseCurve[i] += (float)emor4 * hInv4[i];
    responseCurve[i] += (float)emor5 * hInv5[i];
    if (responseCurve[i] < 0.0f) {
      responseCurve[i] = 0.0f;
    } else if (responseCurve[i] > 1.0f) {
      responseCurve[i] = 1.0f;
    }
  }
  computeInverseResponse();
}
}  // namespace Core
}  // namespace VideoStitch
