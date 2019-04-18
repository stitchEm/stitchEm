// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "sgm.hpp"

#include <libvideostitch/logging.hpp>

#ifndef __ANDROID__
#include "opencv2/core/hal/interface.h"
#else
#include "opencv2/hal/intrin.hpp"
#endif
#include <limits.h>

#if CV_SIMD128
#undef CV_SIMD128
#define CV_SIMD128 0
#endif

namespace VideoStitch {
namespace Core {
namespace SGM {

void aggregateDisparityVolumeSGM(const cv::Mat& costVolume, const VideoStitch::Core::Rect& rect, cv::Mat& disparity,
                                 cv::Mat& buffer, int minDisparity, int numDisparities, int P1, int P2,
                                 int uniquenessRatio, const SGMmode mode) {
#if CV_SIMD128
  // maxDisparity is supposed to multiple of 16, so we can forget doing else
  static const uchar LSBTab[] = {
      0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2,
      0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0,
      1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1,
      0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0,
      2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3,
      0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
      1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};
  static const cv::v_uint16x8 v_LSB(0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80);

  const bool useSIMD = cv::hasSIMD128() &&
                       (minDisparity + numDisparities) % 16 == 0 /* maxDisparity multiple of 16 */ &&
                       ((size_t)costVolume.ptr()) % 16 == 0 /* costVolume is aligned on 16 */;

  if (useSIMD) {
    Logger::get(Logger::Info) << "useSIMD is true" << std::endl;
  }
#endif

  CV_Assert(costVolume.type() == cv::DataType<CostType>::type);
  CV_Assert(disparity.type() == cv::DataType<DispType>::type);

  const int ALIGN = 16;
  const CostType MAX_COST = std::numeric_limits<CostType>::max();

  int minD = minDisparity, maxD = minD + numDisparities;

  // default values if 0s are passed
  uniquenessRatio = uniquenessRatio >= 0 ? uniquenessRatio : 0;
  P1 = P1 > 0 ? P1 : 2;
  P2 = std::max(P2 > 0 ? P2 : 5, P1 + 1);

  int k, width = disparity.cols, height = disparity.rows;
  int D = maxD - minD;
  int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP * DISP_SCALE;
  bool fullDP = mode == SGMmode::SGM_8DIRS;
  int npasses = fullDP ? 2 : 1;

  // NR - the number of directions. the loop on x below that computes Lr assumes that NR == 8.
  // if you change NR, please, modify the loop as well.
  int D2 = D + 16, NRD2 = NR2 * D2;

  // the number of L_r(.,.) and min_k L_r(.,.) lines in the buffer:
  // for 8-way dynamic programming we need the current row and
  // the previous row, i.e. 2 rows in total
  const int NLR = 2;
  const int LrBorder = NLR - 1;

  // for each possible stereo match (img1(x,y) <=> img2(x-d,y))
  // we keep pixel difference cost (C) and the summary cost over NR directions (S).
  // we also keep all the partial costs for the previous line L_r(x,d) and also min_k L_r(x, k)
  const size_t costBufSize = width * D;
  size_t CSBufSize = costBufSize * (fullDP ? height : 1);
  const size_t minLrSize = (width + LrBorder * 2) * NR2, LrSize = minLrSize * D2;
  const size_t totalBufSize = (LrSize + minLrSize) * NLR * sizeof(CostType) +  // minLr[] and Lr[]
                              CSBufSize * sizeof(CostType) +                   // S
                              +16;                                             // for later pointer alignment

  if (buffer.empty() || !buffer.isContinuous() || buffer.cols * buffer.rows * buffer.elemSize() < totalBufSize)
    buffer.create(1, (int)totalBufSize, CV_8U);

  // summary cost over different (nDirs) directions
  CostType* Cbuf = (CostType*)costVolume.ptr();
  CV_Assert(costVolume.isContinuous());
  CostType* Sbuf = (CostType*)cv::alignPtr(buffer.ptr(), ALIGN);

  // add P2 to every C(x,y). it saves a few operations in the inner loops
  for (k = 0; k < (int)costBufSize * height; k++) {
    Cbuf[k] += (CostType)P2;
  }

  for (int pass = 1; pass <= npasses; pass++) {
    int x1, y1, x2, y2, dx, dy;

    if (pass == 1) {
      y1 = (int)rect.top();
      y2 = (int)rect.bottom() + 1;
      dy = 1;
      x1 = (int)rect.left();
      x2 = (int)rect.right() + 1;
      dx = 1;
    } else {
      y1 = (int)rect.bottom();
      y2 = (int)rect.top() - 1;
      dy = -1;
      x1 = (int)rect.right();
      x2 = (int)rect.left() - 1;
      dx = -1;
    }

    CostType *Lr[NLR] = {0}, *minLr[NLR] = {0};

    for (k = 0; k < NLR; k++) {
      // shift Lr[k] and minLr[k] pointers, because we allocated them with the borders,
      // and will occasionally use negative indices with the arrays
      // we need to shift Lr[k] pointers by 1, to give the space for d=-1.
      // however, then the alignment will be imperfect, i.e. bad for SSE,
      // thus we shift the pointers by 8 (8*sizeof(int16_t) == 16 - ideal alignment)
      Lr[k] = Sbuf + CSBufSize + LrSize * k + NRD2 * LrBorder + 8;
      memset(Lr[k] - LrBorder * NRD2 - 8, 0, LrSize * sizeof(CostType));
      minLr[k] = Sbuf + CSBufSize + LrSize * NLR + minLrSize * k + NR2 * LrBorder;
      memset(minLr[k] - LrBorder * NR2, 0, minLrSize * sizeof(CostType));
    }

    for (int y = y1; y != y2; y += dy) {
      int x, d;
      DispType* disparityPtr = disparity.ptr<DispType>(y);
      CostType* C = Cbuf + y * costBufSize;
      CostType* S = Sbuf + (!fullDP ? 0 : y * costBufSize);

      if (pass == 1) {
        // clear the S buffer
        memset(S, 0, width * D * sizeof(S[0]));
      }

      // clear the left and the right borders
      memset(Lr[0] - NRD2 * LrBorder - 8, 0, NRD2 * LrBorder * sizeof(CostType));
      memset(Lr[0] + width * NRD2 - 8, 0, NRD2 * LrBorder * sizeof(CostType));
      memset(minLr[0] - NR2 * LrBorder, 0, NR2 * LrBorder * sizeof(CostType));
      memset(minLr[0] + width * NR2, 0, NR2 * LrBorder * sizeof(CostType));

      /*
       [formula 13 in the paper]
       compute L_r(p, d) = C(p, d) +
       min(L_r(p-r, d),
       L_r(p-r, d-1) + P1,
       L_r(p-r, d+1) + P1,
       min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k)
       where p = (x,y), r is one of the directions.
       we process all the directions at once:
       0: r=(-dx, 0)
       1: r=(-1, -dy)
       2: r=(0, -dy)
       3: r=(1, -dy)
       4: r=(-2, -dy)
       5: r=(-1, -dy*2)
       6: r=(1, -dy*2)
       7: r=(2, -dy)
       */
      for (x = x1; x != x2; x += dx) {
        int xm = x * NR2, xd = xm * D2;

        int delta0 = minLr[0][xm - dx * NR2] + P2, delta1 = minLr[1][xm - NR2 + 1] + P2;
        int delta2 = minLr[1][xm + 2] + P2, delta3 = minLr[1][xm + NR2 + 3] + P2;

        CostType* Lr_p0 = Lr[0] + xd - dx * NRD2;
        CostType* Lr_p1 = Lr[1] + xd - NRD2 + D2;
        CostType* Lr_p2 = Lr[1] + xd + D2 * 2;
        CostType* Lr_p3 = Lr[1] + xd + NRD2 + D2 * 3;

        Lr_p0[-1] = Lr_p0[D] = Lr_p1[-1] = Lr_p1[D] = Lr_p2[-1] = Lr_p2[D] = Lr_p3[-1] = Lr_p3[D] = MAX_COST;

        CostType* Lr_p = Lr[0] + xd;
        const CostType* Cp = C + x * D;
        CostType* Sp = S + x * D;

#if CV_SIMD128
        if (useSIMD) {
          cv::v_int16x8 _P1 = cv::v_setall_s16((int16_t)P1);

          cv::v_int16x8 _delta0 = cv::v_setall_s16((int16_t)delta0);
          cv::v_int16x8 _delta1 = cv::v_setall_s16((int16_t)delta1);
          cv::v_int16x8 _delta2 = cv::v_setall_s16((int16_t)delta2);
          cv::v_int16x8 _delta3 = cv::v_setall_s16((int16_t)delta3);
          cv::v_int16x8 _minL0 = cv::v_setall_s16((int16_t)MAX_COST);

          for (d = 0; d < D; d += 8) {
            cv::v_int16x8 Cpd = cv::v_load(Cp + d);
            cv::v_int16x8 L0, L1, L2, L3;

            L0 = cv::v_load(Lr_p0 + d);
            L1 = cv::v_load(Lr_p1 + d);
            L2 = cv::v_load(Lr_p2 + d);
            L3 = cv::v_load(Lr_p3 + d);

            L0 = cv::v_min(L0, (cv::v_load(Lr_p0 + d - 1) + _P1));
            L0 = cv::v_min(L0, (cv::v_load(Lr_p0 + d + 1) + _P1));

            L1 = cv::v_min(L1, (cv::v_load(Lr_p1 + d - 1) + _P1));
            L1 = cv::v_min(L1, (cv::v_load(Lr_p1 + d + 1) + _P1));

            L2 = cv::v_min(L2, (cv::v_load(Lr_p2 + d - 1) + _P1));
            L2 = cv::v_min(L2, (cv::v_load(Lr_p2 + d + 1) + _P1));

            L3 = cv::v_min(L3, (cv::v_load(Lr_p3 + d - 1) + _P1));
            L3 = cv::v_min(L3, (cv::v_load(Lr_p3 + d + 1) + _P1));

            L0 = cv::v_min(L0, _delta0);
            L0 = ((L0 - _delta0) + Cpd);

            L1 = cv::v_min(L1, _delta1);
            L1 = ((L1 - _delta1) + Cpd);

            L2 = cv::v_min(L2, _delta2);
            L2 = ((L2 - _delta2) + Cpd);

            L3 = cv::v_min(L3, _delta3);
            L3 = ((L3 - _delta3) + Cpd);

            cv::v_store(Lr_p + d, L0);
            cv::v_store(Lr_p + d + D2, L1);
            cv::v_store(Lr_p + d + D2 * 2, L2);
            cv::v_store(Lr_p + d + D2 * 3, L3);

            // Get minimum from in L0-L3
            cv::v_int16x8 t02L, t02H, t13L, t13H, t0123L, t0123H;
            cv::v_zip(L0, L2, t02L, t02H);              // L0[0] L2[0] L0[1] L2[1]...
            cv::v_zip(L1, L3, t13L, t13H);              // L1[0] L3[0] L1[1] L3[1]...
            cv::v_int16x8 t02 = cv::v_min(t02L, t02H);  // L0[i] L2[i] L0[i] L2[i]...
            cv::v_int16x8 t13 = cv::v_min(t13L, t13H);  // L1[i] L3[i] L1[i] L3[i]...
            cv::v_zip(t02, t13, t0123L, t0123H);        // L0[i] L1[i] L2[i] L3[i]...
            cv::v_int16x8 t0 = cv::v_min(t0123L, t0123H);
            _minL0 = cv::v_min(_minL0, t0);

            cv::v_int16x8 Sval = cv::v_load(Sp + d);

            L0 = L0 + L1;
            L2 = L2 + L3;
            Sval = Sval + L0;
            Sval = Sval + L2;

            cv::v_store(Sp + d, Sval);
          }

          cv::v_int32x4 minL, minH;
          cv::v_expand(_minL0, minL, minH);
          cv::v_pack_store(&minLr[0][xm], cv::v_min(minL, minH));
        } else
#endif
        {
          int minL0 = MAX_COST, minL1 = MAX_COST, minL2 = MAX_COST, minL3 = MAX_COST;

          for (d = 0; d < D; d++) {
            int Cpd = Cp[d], L0, L1, L2, L3;

            L0 = Cpd + std::min((int)Lr_p0[d], std::min(Lr_p0[d - 1] + P1, std::min(Lr_p0[d + 1] + P1, delta0))) -
                 delta0;
            L1 = Cpd + std::min((int)Lr_p1[d], std::min(Lr_p1[d - 1] + P1, std::min(Lr_p1[d + 1] + P1, delta1))) -
                 delta1;
            L2 = Cpd + std::min((int)Lr_p2[d], std::min(Lr_p2[d - 1] + P1, std::min(Lr_p2[d + 1] + P1, delta2))) -
                 delta2;
            L3 = Cpd + std::min((int)Lr_p3[d], std::min(Lr_p3[d - 1] + P1, std::min(Lr_p3[d + 1] + P1, delta3))) -
                 delta3;

            Lr_p[d] = (CostType)L0;
            minL0 = std::min(minL0, L0);

            Lr_p[d + D2] = (CostType)L1;
            minL1 = std::min(minL1, L1);

            Lr_p[d + D2 * 2] = (CostType)L2;
            minL2 = std::min(minL2, L2);

            Lr_p[d + D2 * 3] = (CostType)L3;
            minL3 = std::min(minL3, L3);

            Sp[d] = cv::saturate_cast<CostType>(Sp[d] + L0 + L1 + L2 + L3);
          }
          minLr[0][xm] = (CostType)minL0;
          minLr[0][xm + 1] = (CostType)minL1;
          minLr[0][xm + 2] = (CostType)minL2;
          minLr[0][xm + 3] = (CostType)minL3;
        }
      }

      if (pass == npasses) {
        for (x = 0; x < width; x++) {
          disparityPtr[x] = (DispType)INVALID_DISP_SCALED;
        }

        for (x = (int)rect.right(); x >= (int)rect.left(); x--) {
          CostType* Sp = S + x * D;
          int minS = MAX_COST, bestDisp = -1;

          if (npasses == 1) {
            int xm = x * NR2, xd = xm * D2;

            int minL0 = MAX_COST;
            int delta0 = minLr[0][xm + NR2] + P2;
            CostType* Lr_p0 = Lr[0] + xd + NRD2;
            Lr_p0[-1] = Lr_p0[D] = MAX_COST;
            CostType* Lr_p = Lr[0] + xd;

            const CostType* Cp = C + x * D;

#if CV_SIMD128
            if (useSIMD) {
              cv::v_int16x8 _P1 = cv::v_setall_s16((int16_t)P1);
              cv::v_int16x8 _delta0 = cv::v_setall_s16((int16_t)delta0);

              cv::v_int16x8 _minL0 = cv::v_setall_s16((int16_t)minL0);
              cv::v_int16x8 _minS = cv::v_setall_s16(MAX_COST), _bestDisp = cv::v_setall_s16(-1);
              cv::v_int16x8 _d8 = cv::v_int16x8(0, 1, 2, 3, 4, 5, 6, 7), _8 = cv::v_setall_s16(8);

              for (d = 0; d < D; d += 8) {
                cv::v_int16x8 Cpd = cv::v_load(Cp + d);
                cv::v_int16x8 L0 = cv::v_load(Lr_p0 + d);

                L0 = cv::v_min(L0, cv::v_load(Lr_p0 + d - 1) + _P1);
                L0 = cv::v_min(L0, cv::v_load(Lr_p0 + d + 1) + _P1);
                L0 = cv::v_min(L0, _delta0);
                L0 = L0 - _delta0 + Cpd;

                cv::v_store(Lr_p + d, L0);
                _minL0 = cv::v_min(_minL0, L0);
                L0 = L0 + cv::v_load(Sp + d);
                cv::v_store(Sp + d, L0);

                cv::v_int16x8 mask = _minS > L0;
                _minS = cv::v_min(_minS, L0);
                _bestDisp = _bestDisp ^ ((_bestDisp ^ _d8) & mask);
                _d8 += _8;
              }
              int16_t bestDispBuf[8];
              cv::v_store(bestDispBuf, _bestDisp);

              cv::v_int32x4 min32L, min32H;
              cv::v_expand(_minL0, min32L, min32H);
              minLr[0][xm] = (CostType)std::min(cv::v_reduce_min(min32L), cv::v_reduce_min(min32H));

              cv::v_expand(_minS, min32L, min32H);
              minS = std::min(cv::v_reduce_min(min32L), cv::v_reduce_min(min32H));

              cv::v_int16x8 ss = cv::v_setall_s16((int16_t)minS);
              cv::v_uint16x8 minMask = cv::v_reinterpret_as_u16(ss == _minS);
              cv::v_uint16x8 minBit = minMask & v_LSB;

              cv::v_uint32x4 minBitL, minBitH;
              cv::v_expand(minBit, minBitL, minBitH);

              int idx = cv::v_reduce_sum(minBitL) + cv::v_reduce_sum(minBitH);
              bestDisp = bestDispBuf[LSBTab[idx]];
            } else
#endif
            {
              for (d = 0; d < D; d++) {
                int L0 = Cp[d] +
                         std::min((int)Lr_p0[d], std::min(Lr_p0[d - 1] + P1, std::min(Lr_p0[d + 1] + P1, delta0))) -
                         delta0;

                Lr_p[d] = (CostType)L0;
                minL0 = std::min(minL0, L0);

                int Sval = Sp[d] = cv::saturate_cast<CostType>(Sp[d] + L0);
                if (Sval < minS) {
                  minS = Sval;
                  bestDisp = d;
                }
              }
              minLr[0][xm] = (CostType)minL0;
            }
          } else {
#if CV_SIMD128
            if (useSIMD) {
              cv::v_int16x8 _minS = cv::v_setall_s16(MAX_COST), _bestDisp = cv::v_setall_s16(-1);
              cv::v_int16x8 _d8 = cv::v_int16x8(0, 1, 2, 3, 4, 5, 6, 7), _8 = cv::v_setall_s16(8);

              for (d = 0; d < D; d += 8) {
                cv::v_int16x8 L0 = cv::v_load(Sp + d);
                cv::v_int16x8 mask = L0 < _minS;
                _minS = cv::v_min(L0, _minS);
                _bestDisp = _bestDisp ^ ((_bestDisp ^ _d8) & mask);
                _d8 = _d8 + _8;
              }
              cv::v_int32x4 _d0, _d1;
              cv::v_expand(_minS, _d0, _d1);
              minS = (int)std::min(cv::v_reduce_min(_d0), cv::v_reduce_min(_d1));
              cv::v_int16x8 v_mask = cv::v_setall_s16((int16_t)minS) == _minS;

              _bestDisp = (_bestDisp & v_mask) | (cv::v_setall_s16(SHRT_MAX) & ~v_mask);
              cv::v_expand(_bestDisp, _d0, _d1);
              bestDisp = (int)std::min(cv::v_reduce_min(_d0), cv::v_reduce_min(_d1));
            } else
#endif
            {
              for (d = 0; d < D; d++) {
                int Sval = Sp[d];
                if (Sval < minS) {
                  minS = Sval;
                  bestDisp = d;
                }
              }
            }
          }

          for (d = 0; d < D; d++) {
            if (Sp[d] * (100 - uniquenessRatio) < minS * 100 && std::abs(bestDisp - d) > 1) break;
          }
          if (d < D) continue;
          d = bestDisp;

          if (0 < d && d < D - 1) {
            // do subpixel quadratic interpolation:
            //   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
            //   then find minimum of the parabola.
            int denom2 = std::max(Sp[d - 1] + Sp[d + 1] - 2 * Sp[d], 1);
            d = d * DISP_SCALE + ((Sp[d - 1] - Sp[d + 1]) * DISP_SCALE + denom2) / (denom2 * 2);
          } else
            d *= DISP_SCALE;
          disparityPtr[x] = (DispType)(d + minD * DISP_SCALE);
        }
      }

      // now shift the cyclic buffers
      std::swap(Lr[0], Lr[1]);
      std::swap(minLr[0], minLr[1]);
    }
  }
}

template <class saliencyType>
void aggregateDisparityVolumeWithAdaptiveP2SGM(const cv::Mat& saliency, const cv::Mat& costVolume,
                                               const VideoStitch::Core::Rect& rect, cv::Mat& disparity, cv::Mat& buffer,
                                               int minDisparity, int numDisparities, int P1, float P2Alpha, int P2Gamma,
                                               int P2Min, int uniquenessRatio, bool subPixelRefinement,
                                               const SGMmode mode) {
#if CV_SIMD128
  // maxDisparity is supposed to multiple of 16, so we can forget doing else
  static const uchar LSBTab[] = {
      0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2,
      0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0,
      1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1,
      0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0,
      2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3,
      0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
      1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};
  static const cv::v_uint16x8 v_LSB(0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80);

  const bool useSIMD = cv::hasSIMD128() &&
                       (minDisparity + numDisparities) % 16 == 0 /* maxDisparity multiple of 16 */ &&
                       ((size_t)costVolume.ptr()) % 16 == 0 /* costVolume is aligned on 16 */;

  if (useSIMD) {
    Logger::get(Logger::Info) << "useSIMD is true" << std::endl;
  }
#endif

  // CV_Assert(saliency.type() == cv::DataType<saliencyType>::type);
  // CV_Assert(costVolume.type() == cv::DataType<CostType>::type);
  // CV_Assert(disparity.type() == cv::DataType<DispType>::type);
  // CV_Assert(saliency.rows == rect.getHeight() && saliency.cols == rect.getWidth());

  const int ALIGN = 16;
  const CostType MAX_COST = std::numeric_limits<CostType>::max();

  int minD = minDisparity, maxD = minD + numDisparities;

  // default values if 0s are passed
  uniquenessRatio = uniquenessRatio >= 0 ? uniquenessRatio : 0;
  P1 = P1 > 0 ? P1 : 2;

  int k, width = disparity.cols, height = disparity.rows;
  int D = maxD - minD;
  int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP * DISP_SCALE;
  bool fullDP = mode == SGMmode::SGM_8DIRS;
  int npasses = fullDP ? 2 : 1;

  // NR - the number of directions. the loop on x below that computes Lr assumes that NR == 8.
  // if you change NR, please, modify the loop as well.
  int D2 = D + 16, NRD2 = NR2 * D2;

  // the number of L_r(.,.) and min_k L_r(.,.) lines in the buffer:
  // for 8-way dynamic programming we need the current row and
  // the previous row, i.e. 2 rows in total
  const int NLR = 2;
  const int LrBorder = NLR - 1;

  // for each possible stereo match (img1(x,y) <=> img2(x-d,y))
  // we keep pixel difference cost (C) and the summary cost over NR directions (S).
  // we also keep all the partial costs for the previous line L_r(x,d) and also min_k L_r(x, k)
  const size_t costBufSize = width * D;
  size_t CSBufSize = costBufSize * (fullDP ? height : 1);
  const size_t minLrSize = (width + LrBorder * 2) * NR2, LrSize = minLrSize * D2;
  const size_t totalBufSize = (LrSize + minLrSize) * NLR * sizeof(CostType) +  // minLr[] and Lr[]
                              CSBufSize * sizeof(CostType) +                   // S
                              +16;                                             // for later pointer alignment

  if (buffer.empty() || !buffer.isContinuous() || buffer.cols * buffer.rows * buffer.elemSize() < totalBufSize)
    buffer.create(1, (int)totalBufSize, CV_8U);

  // summary cost over different (nDirs) directions
  CostType* Cbuf = (CostType*)costVolume.ptr();
  CV_Assert(costVolume.isContinuous());
  CostType* Sbuf = (CostType*)cv::alignPtr(buffer.ptr(), ALIGN);

  for (int pass = 1; pass <= npasses; pass++) {
    int x1, y1, x2, y2, dx, dy;

    if (pass == 1) {
      y1 = (int)rect.top();
      y2 = (int)rect.bottom() + 1;
      dy = 1;
      x1 = (int)rect.left();
      x2 = (int)rect.right() + 1;
      dx = 1;
    } else {
      y1 = (int)rect.bottom();
      y2 = (int)rect.top() - 1;
      dy = -1;
      x1 = (int)rect.right();
      x2 = (int)rect.left() - 1;
      dx = -1;
    }

    CostType *Lr[NLR] = {0}, *minLr[NLR] = {0};

    for (k = 0; k < NLR; k++) {
      // shift Lr[k] and minLr[k] pointers, because we allocated them with the borders,
      // and will occasionally use negative indices with the arrays
      // we need to shift Lr[k] pointers by 1, to give the space for d=-1.
      // however, then the alignment will be imperfect, i.e. bad for SSE,
      // thus we shift the pointers by 8 (8*sizeof(int16_t) == 16 - ideal alignment)
      Lr[k] = Sbuf + CSBufSize + LrSize * k + NRD2 * LrBorder + 8;
      memset(Lr[k] - LrBorder * NRD2 - 8, 0, LrSize * sizeof(CostType));
      minLr[k] = Sbuf + CSBufSize + LrSize * NLR + minLrSize * k + NR2 * LrBorder;
      memset(minLr[k] - LrBorder * NR2, 0, minLrSize * sizeof(CostType));
    }

    // saliency for current volume position
    saliencyType saliencyCurrent;

    auto adaptP2 = [&](int x, int y) -> int {
      saliencyType saliencyAtXY = rect.contains(x, y)
                                      ? saliency.at<saliencyType>((int)(y - rect.top()), (int)(x - rect.left()))
                                      : saliencyCurrent;

      return std::max(P2Min, int(-P2Alpha * cv::norm(cv::Vec<short, saliencyType::channels>(saliencyAtXY) -
                                                         cv::Vec<short, saliencyType::channels>(saliencyCurrent),
                                                     cv::NORM_L1) +
                                 P2Gamma));
    };

    for (int y = y1; y != y2; y += dy) {
      int x, d;
      DispType* const disparityPtr = disparity.ptr<DispType>(y);
      const CostType* const C = Cbuf + y * costBufSize;
      CostType* const S = Sbuf + (!fullDP ? 0 : y * costBufSize);

      if (pass == 1) {
        // clear the S buffer
        memset(S, 0, width * D * sizeof(S[0]));
      }

      // clear the left and the right borders
      memset(Lr[0] - NRD2 * LrBorder - 8, 0, NRD2 * LrBorder * sizeof(CostType));
      memset(Lr[0] + width * NRD2 - 8, 0, NRD2 * LrBorder * sizeof(CostType));
      memset(minLr[0] - NR2 * LrBorder, 0, NR2 * LrBorder * sizeof(CostType));
      memset(minLr[0] + width * NR2, 0, NR2 * LrBorder * sizeof(CostType));

      /*
       [formula 13 in the paper]
       compute L_r(p, d) = C(p, d) +
       min(L_r(p-r, d),
       L_r(p-r, d-1) + P1,
       L_r(p-r, d+1) + P1,
       min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k)
       where p = (x,y), r is one of the directions.
       we process all the directions at once:
       0: r=(-dx, 0)
       1: r=(-1, -dy)
       2: r=(0, -dy)
       3: r=(1, -dy)
       4: r=(-2, -dy)
       5: r=(-1, -dy*2)
       6: r=(1, -dy*2)
       7: r=(2, -dy)

       Note: the code below uses only the first four paths, but enough memory was allocated to use the 8 paths
       */
      for (x = x1; x != x2; x += dx) {
        const int xm = x * NR2, xd = xm * D2;

        int delta0 = minLr[0][xm - dx * NR2];  // minLr[0][(x - dx) * NR2]
        int delta1 = minLr[1][xm - NR2 + 1];   // minLr[1][(x -  1) * NR2 + 1]
        int delta2 = minLr[1][xm + 2];         // minLr[1][(x     ) * NR2 + 2]
        int delta3 = minLr[1][xm + NR2 + 3];   // minLr[1][(x +  1) * NR2 + 3]

        // get saliency for current position
        saliencyCurrent = saliency.at<saliencyType>((int)(y - rect.top()), (int)(x - rect.left()));

        const int L0_P2 = adaptP2(x - dx, y);
        const int L1_P2 = adaptP2(x - dx, y - dy);
        const int L2_P2 = adaptP2(x, y - dy);
        const int L3_P2 = adaptP2(x + dx, y - dy);

        CostType* const Lr_p0 = Lr[0] + xd - dx * NRD2;      // Lr[0] + (x - dx) * NRD2
        CostType* const Lr_p1 = Lr[1] + xd - NRD2 + D2;      // Lr[1] + (x -  1) * NRD2 + D2
        CostType* const Lr_p2 = Lr[1] + xd + D2 * 2;         // Lr[1] + (x     ) * NRD2 + D2 * 2
        CostType* const Lr_p3 = Lr[1] + xd + NRD2 + D2 * 3;  // Lr[1] + (x +  1) * NRD2 + D2 * 3

        Lr_p0[-1] = Lr_p0[D] = Lr_p1[-1] = Lr_p1[D] = Lr_p2[-1] = Lr_p2[D] = Lr_p3[-1] = Lr_p3[D] = MAX_COST;

        CostType* const Lr_p = Lr[0] + xd;  // Lr[0] + x * NRD2
        const CostType* const Cp = C + x * D;
        CostType* const Sp = S + x * D;

#if CV_SIMD128
        if (useSIMD) {
          cv::v_int16x8 _P1 = cv::v_setall_s16((int16_t)P1);

          cv::v_int16x8 _L0_P2 = cv::v_setall_s16((int16_t)L0_P2);
          cv::v_int16x8 _L1_P2 = cv::v_setall_s16((int16_t)L1_P2);
          cv::v_int16x8 _L2_P2 = cv::v_setall_s16((int16_t)L2_P2);
          cv::v_int16x8 _L3_P2 = cv::v_setall_s16((int16_t)L3_P2);
          cv::v_int16x8 _delta0 = cv::v_setall_s16((int16_t)delta0);
          cv::v_int16x8 _delta1 = cv::v_setall_s16((int16_t)delta1);
          cv::v_int16x8 _delta2 = cv::v_setall_s16((int16_t)delta2);
          cv::v_int16x8 _delta3 = cv::v_setall_s16((int16_t)delta3);
          cv::v_int16x8 _minL0 = cv::v_setall_s16((int16_t)MAX_COST);

          for (d = 0; d < D; d += 8) {
            cv::v_int16x8 Cpd = cv::v_load(Cp + d);
            cv::v_int16x8 L0, L1, L2, L3;

            L0 = cv::v_load(Lr_p0 + d);
            L1 = cv::v_load(Lr_p1 + d);
            L2 = cv::v_load(Lr_p2 + d);
            L3 = cv::v_load(Lr_p3 + d);

            L0 = cv::v_min(L0, (cv::v_load(Lr_p0 + d - 1) + _P1));
            L0 = cv::v_min(L0, (cv::v_load(Lr_p0 + d + 1) + _P1));

            L1 = cv::v_min(L1, (cv::v_load(Lr_p1 + d - 1) + _P1));
            L1 = cv::v_min(L1, (cv::v_load(Lr_p1 + d + 1) + _P1));

            L2 = cv::v_min(L2, (cv::v_load(Lr_p2 + d - 1) + _P1));
            L2 = cv::v_min(L2, (cv::v_load(Lr_p2 + d + 1) + _P1));

            L3 = cv::v_min(L3, (cv::v_load(Lr_p3 + d - 1) + _P1));
            L3 = cv::v_min(L3, (cv::v_load(Lr_p3 + d + 1) + _P1));

            L0 = cv::v_min(L0, _delta0 + _L0_P2);
            L0 = ((L0 - _delta0) + Cpd);

            L1 = cv::v_min(L1, _delta1 + _L1_P2);
            L1 = ((L1 - _delta1) + Cpd);

            L2 = cv::v_min(L2, _delta2 + _L2_P2);
            L2 = ((L2 - _delta2) + Cpd);

            L3 = cv::v_min(L3, _delta3 + _L3_P2);
            L3 = ((L3 - _delta3) + Cpd);

            cv::v_store(Lr_p + d, L0);
            cv::v_store(Lr_p + d + D2, L1);
            cv::v_store(Lr_p + d + D2 * 2, L2);
            cv::v_store(Lr_p + d + D2 * 3, L3);

            // Get minimum from in L0-L3
            cv::v_int16x8 t02L, t02H, t13L, t13H, t0123L, t0123H;
            cv::v_zip(L0, L2, t02L, t02H);              // L0[0] L2[0] L0[1] L2[1]...
            cv::v_zip(L1, L3, t13L, t13H);              // L1[0] L3[0] L1[1] L3[1]...
            cv::v_int16x8 t02 = cv::v_min(t02L, t02H);  // L0[i] L2[i] L0[i] L2[i]...
            cv::v_int16x8 t13 = cv::v_min(t13L, t13H);  // L1[i] L3[i] L1[i] L3[i]...
            cv::v_zip(t02, t13, t0123L, t0123H);        // L0[i] L1[i] L2[i] L3[i]...
            cv::v_int16x8 t0 = cv::v_min(t0123L, t0123H);
            _minL0 = cv::v_min(_minL0, t0);

            cv::v_int16x8 Sval = cv::v_load(Sp + d);

            L0 = L0 + L1;
            L2 = L2 + L3;
            Sval = Sval + L0;
            Sval = Sval + L2;

            cv::v_store(Sp + d, Sval);
          }

          cv::v_int32x4 minL, minH;
          cv::v_expand(_minL0, minL, minH);
          cv::v_pack_store(&minLr[0][xm], cv::v_min(minL, minH));
        } else
#endif
        {
          int minL0 = MAX_COST, minL1 = MAX_COST, minL2 = MAX_COST, minL3 = MAX_COST;

          for (d = 0; d < D; d++) {
            int Cpd = Cp[d], L0, L1, L2, L3;

            L0 = Cpd +
                 std::min((int)Lr_p0[d], std::min(Lr_p0[d - 1] + P1, std::min(Lr_p0[d + 1] + P1, delta0 + L0_P2))) -
                 delta0;
            L1 = Cpd +
                 std::min((int)Lr_p1[d], std::min(Lr_p1[d - 1] + P1, std::min(Lr_p1[d + 1] + P1, delta1 + L1_P2))) -
                 delta1;
            L2 = Cpd +
                 std::min((int)Lr_p2[d], std::min(Lr_p2[d - 1] + P1, std::min(Lr_p2[d + 1] + P1, delta2 + L2_P2))) -
                 delta2;
            L3 = Cpd +
                 std::min((int)Lr_p3[d], std::min(Lr_p3[d - 1] + P1, std::min(Lr_p3[d + 1] + P1, delta3 + L3_P2))) -
                 delta3;

            Lr_p[d] = (CostType)L0;
            minL0 = std::min(minL0, L0);

            Lr_p[d + D2] = (CostType)L1;
            minL1 = std::min(minL1, L1);

            Lr_p[d + D2 * 2] = (CostType)L2;
            minL2 = std::min(minL2, L2);

            Lr_p[d + D2 * 3] = (CostType)L3;
            minL3 = std::min(minL3, L3);

            Sp[d] = cv::saturate_cast<CostType>(Sp[d] + L0 + L1 + L2 + L3);
          }
          minLr[0][xm] = (CostType)minL0;
          minLr[0][xm + 1] = (CostType)minL1;
          minLr[0][xm + 2] = (CostType)minL2;
          minLr[0][xm + 3] = (CostType)minL3;
        }
      }

      if (pass == npasses) {
        for (x = 0; x < width; x++) {
          disparityPtr[x] = (DispType)INVALID_DISP_SCALED;
        }

        for (x = (int)rect.right(); x >= (int)rect.left(); x--) {
          CostType* Sp = S + x * D;
          int minS = MAX_COST, bestDisp = -1;

          if (npasses == 1) {
            int xm = x * NR2, xd = xm * D2;

            int minL0 = MAX_COST;
            const int delta0 = minLr[0][xm + NR2];
            CostType* const Lr_p0 = Lr[0] + xd + NRD2;
            Lr_p0[-1] = Lr_p0[D] = MAX_COST;
            CostType* const Lr_p = Lr[0] + xd;

            const CostType* const Cp = C + x * D;

            saliencyCurrent = saliency.at<saliencyType>((int)(y - rect.top()), (int)(x - rect.left()));

            const int L0_P2 = adaptP2(x + dx, y);

#if CV_SIMD128
            if (useSIMD) {
              cv::v_int16x8 _P2 = cv::v_setall_s16((int16_t)L0_P2);
              cv::v_int16x8 _P1 = cv::v_setall_s16((int16_t)P1);
              cv::v_int16x8 _delta0 = cv::v_setall_s16((int16_t)delta0);

              cv::v_int16x8 _minL0 = cv::v_setall_s16((int16_t)minL0);
              cv::v_int16x8 _minS = cv::v_setall_s16(MAX_COST), _bestDisp = cv::v_setall_s16(-1);
              cv::v_int16x8 _d8 = cv::v_int16x8(0, 1, 2, 3, 4, 5, 6, 7), _8 = cv::v_setall_s16(8);

              for (d = 0; d < D; d += 8) {
                cv::v_int16x8 Cpd = cv::v_load(Cp + d);
                cv::v_int16x8 L0 = cv::v_load(Lr_p0 + d);

                L0 = cv::v_min(L0, cv::v_load(Lr_p0 + d - 1) + _P1);
                L0 = cv::v_min(L0, cv::v_load(Lr_p0 + d + 1) + _P1);
                L0 = cv::v_min(L0, _delta0 + _P2);
                L0 = L0 - _delta0 + Cpd;

                cv::v_store(Lr_p + d, L0);
                _minL0 = cv::v_min(_minL0, L0);
                L0 = L0 + cv::v_load(Sp + d);
                cv::v_store(Sp + d, L0);

                cv::v_int16x8 mask = _minS > L0;
                _minS = cv::v_min(_minS, L0);
                _bestDisp = _bestDisp ^ ((_bestDisp ^ _d8) & mask);
                _d8 += _8;
              }
              int16_t bestDispBuf[8];
              cv::v_store(bestDispBuf, _bestDisp);

              cv::v_int32x4 min32L, min32H;
              cv::v_expand(_minL0, min32L, min32H);
              minLr[0][xm] = (CostType)std::min(cv::v_reduce_min(min32L), cv::v_reduce_min(min32H));

              cv::v_expand(_minS, min32L, min32H);
              minS = std::min(cv::v_reduce_min(min32L), cv::v_reduce_min(min32H));

              cv::v_int16x8 ss = cv::v_setall_s16((int16_t)minS);
              cv::v_uint16x8 minMask = cv::v_reinterpret_as_u16(ss == _minS);
              cv::v_uint16x8 minBit = minMask & v_LSB;

              cv::v_uint32x4 minBitL, minBitH;
              cv::v_expand(minBit, minBitL, minBitH);

              int idx = cv::v_reduce_sum(minBitL) + cv::v_reduce_sum(minBitH);
              bestDisp = bestDispBuf[LSBTab[idx]];
            } else
#endif
            {
              for (d = 0; d < D; d++) {
                const int L0 =
                    Cp[d] +
                    std::min((int)Lr_p0[d], std::min(Lr_p0[d - 1] + P1, std::min(Lr_p0[d + 1] + P1, delta0 + L0_P2))) -
                    delta0;

                Lr_p[d] = (CostType)L0;
                minL0 = std::min(minL0, L0);

                int Sval = Sp[d] = cv::saturate_cast<CostType>(Sp[d] + L0);

                if (Sval < minS) {
                  minS = Sval;
                  bestDisp = d;
                }
              }
              minLr[0][xm] = (CostType)minL0;
            }
          } else {
#if CV_SIMD128
            if (useSIMD) {
              cv::v_int16x8 _minS = cv::v_setall_s16(MAX_COST), _bestDisp = cv::v_setall_s16(-1);
              cv::v_int16x8 _d8 = cv::v_int16x8(0, 1, 2, 3, 4, 5, 6, 7), _8 = cv::v_setall_s16(8);

              for (d = 0; d < D; d += 8) {
                cv::v_int16x8 L0 = cv::v_load(Sp + d);
                cv::v_int16x8 mask = L0 < _minS;
                _minS = cv::v_min(L0, _minS);
                _bestDisp = _bestDisp ^ ((_bestDisp ^ _d8) & mask);
                _d8 = _d8 + _8;
              }
              cv::v_int32x4 _d0, _d1;
              cv::v_expand(_minS, _d0, _d1);
              minS = (int)std::min(cv::v_reduce_min(_d0), cv::v_reduce_min(_d1));
              cv::v_int16x8 v_mask = cv::v_setall_s16((int16_t)minS) == _minS;

              _bestDisp = (_bestDisp & v_mask) | (cv::v_setall_s16(SHRT_MAX) & ~v_mask);
              cv::v_expand(_bestDisp, _d0, _d1);
              bestDisp = (int)std::min(cv::v_reduce_min(_d0), cv::v_reduce_min(_d1));
            } else
#endif
            {
              for (d = 0; d < D; d++) {
                int Sval = Sp[d];
                if (Sval < minS) {
                  minS = Sval;
                  bestDisp = d;
                }
              }
            }
          }

          for (d = 0; d < D; d++) {
            if (Sp[d] * (100 - uniquenessRatio) < minS * 100 && std::abs(bestDisp - d) > 1) break;
          }
          if (d < D) {
            continue;
          }
          d = bestDisp;

          if (subPixelRefinement) {
            if (0 < d && d < D - 1) {
              // do subpixel quadratic interpolation:
              //   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
              //   then find minimum of the parabola.
              int denom2 = std::max(Sp[d - 1] + Sp[d + 1] - 2 * Sp[d], 1);
              d = d * DISP_SCALE + ((Sp[d - 1] - Sp[d + 1]) * DISP_SCALE + denom2) / (denom2 * 2);
            } else {
              d *= DISP_SCALE;
            }
            disparityPtr[x] = (DispType)(d + minD * DISP_SCALE);
          } else {
            disparityPtr[x] = (DispType)(d + minD);
          }
        }
      }

      // now shift the cyclic buffers
      std::swap(Lr[0], Lr[1]);
      std::swap(minLr[0], minLr[1]);
    }
  }
}

template void aggregateDisparityVolumeWithAdaptiveP2SGM<cv::Vec<uchar, 4>>(
    const cv::Mat& saliency, const cv::Mat& costVolume, const VideoStitch::Core::Rect& rect, cv::Mat& disparity,
    cv::Mat& buffer, int minDisparity, int numDisparities, int P1, float P2Alpha, int P2Gamma, int P2Min,
    int uniquenessRatio, bool subPixelRefinement, const SGMmode mode);

template void aggregateDisparityVolumeWithAdaptiveP2SGM<cv::Vec<uchar, 1>>(
    const cv::Mat& saliency, const cv::Mat& costVolume, const VideoStitch::Core::Rect& rect, cv::Mat& disparity,
    cv::Mat& buffer, int minDisparity, int numDisparities, int P1, float P2Alpha, int P2Gamma, int P2Min,
    int uniquenessRatio, bool subPixelRefinement, const SGMmode mode);

}  // namespace SGM
}  // namespace Core
}  // namespace VideoStitch
