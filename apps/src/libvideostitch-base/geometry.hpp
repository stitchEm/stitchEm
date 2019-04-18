// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "common-config.hpp"

#include <qmath.h>
#include <QVector3D>

qreal VS_COMMON_EXPORT radToDeg(qreal v);

qreal VS_COMMON_EXPORT degToRad(qreal v);

void VS_COMMON_EXPORT equirectangular2sphere(QPointF uv, QVector3D& vec, qreal sphereRadius);

void VS_COMMON_EXPORT rectilinear2sphere(QPointF uv, QVector3D& v, qreal sphereRadius);

void VS_COMMON_EXPORT stereographic2sphere(QPointF uv, QVector3D& v, qreal sphereRadius);

void VS_COMMON_EXPORT spherical2sphere(QPointF uv, QVector3D& v, qreal sphereRadius);

void VS_COMMON_EXPORT sphere2orientation(const QVector3D& v1, const QVector3D& v2, qreal& yaw, qreal& pitch,
                                         qreal& roll);

void VS_COMMON_EXPORT quaternion2euler(qreal q0, qreal q1, qreal q2, qreal q3, qreal& yaw, qreal& pitch, qreal& roll);

void VS_COMMON_EXPORT euler2quaternion(qreal yaw, qreal pitch, qreal roll, qreal& q0, qreal& q1, qreal& q2, qreal& q3);

// apply o, then p
void VS_COMMON_EXPORT quaternionProduct(qreal o0, qreal o1, qreal o2, qreal o3, qreal p0, qreal p1, qreal p2, qreal p3,
                                        qreal& q0, qreal& q1, qreal& q2, qreal& q3);
