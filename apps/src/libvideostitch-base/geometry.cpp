// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "geometry.hpp"

qreal radToDeg(qreal v) { return v * (180.0 / M_PI); }

qreal degToRad(qreal deg) { return (qreal)(M_PI * (deg / 180.0)); }

void rectilinear2equirectangular(QPointF& uv, qreal sphereRadius) {
  uv.ry() = qAtan2(uv.ry(), qSqrt(sphereRadius * sphereRadius + uv.rx() * uv.rx()));
  uv.rx() = qAtan2(uv.rx(), sphereRadius);
}

void spherical2equirectangular(QPointF& uv, qreal sphereRadius) {
  uv /= sphereRadius;
  qreal r = qSqrt(uv.rx() * uv.rx() + uv.ry() * uv.ry());
  qreal theta = r;
  qreal s = theta != 0.0f ? qSin(theta) / r : 0.0f;
  QPointF t(qCos(theta), s * uv.rx());
  uv.rx() = qAtan2(t.ry(), t.rx());
  uv.ry() = qAtan2(s * uv.ry(), qSqrt(t.rx() * t.rx() + t.ry() * t.ry()));
}

void stereographic2equirectangular(QPointF& uv, qreal sphereRadius) {
  uv /= sphereRadius;
  qreal rh = qSqrt(uv.rx() * uv.rx() + uv.ry() * uv.ry());
  if (rh > 0.0f) {
    qreal c = 2.0f * qAtan(rh / 2.0f);
    qreal sin_c = qSin(c);
    qreal cos_c = qCos(c);
    uv.rx() = qAtan2(uv.rx() * sin_c, cos_c * rh);
    uv.ry() = qAsin((uv.ry() * sin_c) / rh);
  }
}

void equirectangular2sphere(QPointF uv, QVector3D& vec, qreal sphereRadius) {
  uv /= sphereRadius;
  qreal phi = uv.x();
  qreal theta = -uv.y() + M_PI / 2.0f;
  // Pass above the north pole
  if (theta < 0.0f) {
    theta = -theta;
    phi += M_PI;
  }
  // Pass above the south pole
  if (theta > M_PI) {
    theta = 2.0f * M_PI - theta;
    phi += M_PI;
  }
  qreal sinTheta = qSin(theta);
  QPointF v(sinTheta * qSin(phi), qCos(theta));
  qreal r = qSqrt(v.rx() * v.rx() + v.ry() * v.ry());
  if (r != 0.0f) {
    // Normal case, atan2f is defined in this domain.
    uv = (qAtan2(r, sinTheta * qCos(phi)) / r) * v;
  } else {
    // atan2f is not defined around (0,0) and (pi + k pi, pi/2).
    // The result is taken to be 0 at (0,0) and defined by continuity modulo (2 pi) along the phi axis to be (pi at (pi
    // + k pi, pi/2).
    if (qAbs(phi) < 0.001f) {  // <==> sin(theta) * cos(phi) >= 0
      uv.rx() = 0.0f;
      uv.ry() = 0.0f;
    } else {
      uv.rx() = M_PI;
      uv.ry() = 0.0f;
    }
  }
  theta = qSqrt(uv.rx() * uv.rx() + uv.ry() * uv.ry());
  vec.setX(uv.rx() * qSin(theta) / theta);
  vec.setY(uv.ry() * qSin(theta) / theta);
  vec.setZ(qCos(theta));
}

void rectilinear2sphere(QPointF uv, QVector3D& v, qreal sphereRadius) {
  rectilinear2equirectangular(uv, sphereRadius);
  return equirectangular2sphere(uv, v, 1);
}

void stereographic2sphere(QPointF uv, QVector3D& v, qreal sphereRadius) {
  stereographic2equirectangular(uv, sphereRadius);
  return equirectangular2sphere(uv, v, 1);
}

void spherical2sphere(QPointF uv, QVector3D& v, qreal sphereRadius) {
  spherical2equirectangular(uv, sphereRadius);
  return equirectangular2sphere(uv, v, 1);
}

void sphere2orientation(const QVector3D& v1, const QVector3D& v2, qreal& yaw, qreal& pitch, qreal& roll) {
  qreal dotProduct = qMin(QVector3D::dotProduct(v1, v2), 1.0f);
  qreal theta_2 = qAcos(dotProduct) / 2.0f;
  QVector3D nu = QVector3D::normal(v1, v2);
  qreal sintheta_2 = qSin(theta_2);
  qreal q0 = qCos(theta_2);
  qreal q1 = sintheta_2 * nu.x();
  qreal q2 = sintheta_2 * nu.y();
  qreal q3 = sintheta_2 * nu.z();
  quaternion2euler(q0, q1, q2, q3, yaw, pitch, roll);
}

void quaternion2euler(qreal q0, qreal q1, qreal q2, qreal q3, qreal& yaw, qreal& pitch, qreal& roll) {
  yaw = qAtan2(2.0 * (q1 * q3 - q0 * q2), q3 * q3 - q2 * q2 - q1 * q1 + q0 * q0);
  pitch = -qAsin(2.0 * (q2 * q3 + q0 * q1));
  roll = qAtan2(2.0 * (q1 * q2 - q0 * q3), q2 * q2 - q3 * q3 + q0 * q0 - q1 * q1);
}

void euler2quaternion(qreal yaw, qreal pitch, qreal roll, qreal& q0, qreal& q1, qreal& q2, qreal& q3) {
  qreal cy = qCos(yaw * 0.5);
  qreal cp = qCos(pitch * 0.5);
  qreal cr = qCos(roll * 0.5);
  qreal sy = qSin(yaw * 0.5);
  qreal sp = qSin(pitch * 0.5);
  qreal sr = qSin(roll * 0.5);

  q0 = -cr * cp * cy - sr * sp * sy;
  q1 = cr * cy * sp + sr * cp * sy;
  q2 = cr * cp * sy - sr * cy * sp;
  q3 = -cr * sp * sy + cp * cy * sr;
}

void quaternionProduct(qreal o0, qreal o1, qreal o2, qreal o3, qreal p0, qreal p1, qreal p2, qreal p3, qreal& q0,
                       qreal& q1, qreal& q2, qreal& q3) {
  q0 = o0 * p0 - o1 * p1 - o2 * p2 - o3 * p3;
  q1 = o0 * p1 + o1 * p0 + o2 * p3 - o3 * p2;
  q2 = o0 * p2 - o1 * p3 + o2 * p0 + o3 * p1;
  q3 = o0 * p3 + o1 * p2 - o2 * p1 + o3 * p0;
}
