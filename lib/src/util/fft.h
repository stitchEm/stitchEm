#ifndef FFT_H_
#define FFT_H_

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#define USE_CDFT_WINTHREADS
#else
#define USE_CDFT_PTHREADS
#endif

#define FFT (1)
#define IFFT (-1)

void cdft(int n, int isgn, double *a);

void rdft(int n, int isgn, double *a);

void ddct(int n, int isgn, double *a);

void ddst(int n, int isgn, double *a);

void dfct(int n, double *a);

void dfst(int n, double *a);

#ifdef __cplusplus
}
#endif

#endif
