// poly34.h : solution of cubic and quartic equation
// (c) Khashin S.I. http://math.ivanovo.ac.ru/dalgebra/Khashin/index.html
// khash2 (at) gmail.com
#include "gpu/vectorTypes.hpp"

#undef FN_NAME
#undef INL_FN_FLOAT
#undef INL_FN_FLOAT2
#undef INL_FN_FLOAT3
#undef INL_FN_QUARTIC_SOL

#define FN_NAME(fnName) fnName
#define INL_FN_FLOAT(fnName) inline double fnName
#define INL_FN_FLOAT2(fnName) inline double2 fnName
#define INL_FN_FLOAT3(fnName) inline double3 fnName
#define INL_FN_QUARTIC_SOL(fnName) inline vsQuarticSolution fnName

namespace VideoStitch {
namespace Core {

// solve equation a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 , where a, b, c, d, e could be 0
// details of the output parameters are described in the solveP3 and solveP4
INL_FN_QUARTIC_SOL(solveQuartic)(const double a, const double b, const double c, const double d, const double e);

INL_FN_QUARTIC_SOL(solveP1)(const double a);                  // solve linear equation x + a = 0
INL_FN_QUARTIC_SOL(solveP2)(const double a, const double b);  // solve quadratic equation x^2 + a*x + b = 0
INL_FN_QUARTIC_SOL(solveP3)
(const double a, const double b, const double c);  // solve cubic equation x^3 + a*x^2 + b*x + c = 0
INL_FN_QUARTIC_SOL(solveP4)
(const double a, const double b, const double c,
 const double d);  // solve equation x^4 + a*x^3 + b*x^2 + c*x + d = 0 by Dekart-Euler method
// x - array of size 4
// return 4: 4 real roots x[0], x[1], x[2], x[3], possible multiple roots
// return 2: 2 real roots x[0], x[1] and complex x[2]±i*x[3],
// return 0: two pair of complex roots: x[0]±i*x[1],  x[2]±i*x[3],

INL_FN_QUARTIC_SOL(solveP4Bi)(const double b, const double d);  // solve equation x^4 + b*x^2 + d = 0
INL_FN_QUARTIC_SOL(solveP4De)
(const double b, const double c, const double d);      // solve equation x^4 + b*x^2 + c*x + d = 0
INL_FN_FLOAT2(cSqrt)(const double x, const double y);  // returns as a+i*s, sqrt(x+i*y)
INL_FN_FLOAT(n4Step)
(const double x, const double a, const double b, const double c,
 const double d);  // one Newton step for x^4 + a*x^3 + b*x^2 + c*x + d
INL_FN_QUARTIC_SOL(setQuarticSolution)(const int solutionCount, const double x[4]);

}  // namespace Core
}  // namespace VideoStitch
