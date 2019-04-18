/****************************************************************************
**
** Copyright (C) 2015 The Qt Company Ltd.
** Contact: http://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "levelmeter.hpp"
#include "libvideostitch/logging.hpp"

#include <math.h>

#include <QPainter>
#include <QTimer>
#include <QDebug>
#include <qmath.h>
#include <QtGlobal>

// Constants
const int RedrawInterval = 100;  // ms
const qreal PeakDecayRate = 0.001;
const int PeakHoldLevelDuration = 2000;  // ms
const std::string tag("levelMeter");

#define dBToLin(x) qPow(10, x / 20.)

OneLevelMeter::OneLevelMeter(QWidget *parent)
    : QWidget(parent),
      m_rmsLevel(0.0),
      m_peakLevel(0.0),
      m_decayedPeakLevel(0.0),
      m_peakDecayRate(PeakDecayRate),
      m_peakHoldLevel(0.0),
      m_redrawTimer(new QTimer(this)),
      m_rmsColor(255, 255, 255),
      m_peakColor(m_rmsColor) {
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
  setMinimumWidth(10);

  connect(m_redrawTimer, &QTimer::timeout, this, &OneLevelMeter::redrawTimerExpired);

  m_redrawTimer->start(RedrawInterval);
}

OneLevelMeter::~OneLevelMeter() {}

void OneLevelMeter::reset() {
  m_rmsLevel = 0.0;
  m_peakLevel = 0.0;
  update();
}

void OneLevelMeter::levelChanged(qreal rmsLevel, qreal peakLevel) {
  qreal peakLevelLin = dBToLin(peakLevel);
  m_rmsLevel = dBToLin(rmsLevel);
  if (peakLevelLin > m_decayedPeakLevel) {
    m_peakLevel = peakLevelLin;
    m_decayedPeakLevel = peakLevelLin;
    m_peakLevelChanged.start();
  }

  if (peakLevelLin > m_peakHoldLevel) {
    m_peakHoldLevel = peakLevelLin;
    m_peakHoldLevelChanged.start();
  }

  update();
}

void OneLevelMeter::redrawTimerExpired() {
  // Decay the peak signal
  const int elapsedMs = m_peakLevelChanged.elapsed();
  const qreal decayAmount = m_peakDecayRate * elapsedMs;
  m_decayedPeakLevel = qMax(m_peakLevel - decayAmount, 0.0);

  // Check whether to clear the peak hold level
  if (m_peakHoldLevelChanged.elapsed() > PeakHoldLevelDuration) {
    m_peakHoldLevel = 0.0;
  }
  update();
}

void OneLevelMeter::paintEvent(QPaintEvent *event) {
  Q_UNUSED(event)

  QPainter painter(this);
  // painter.fillRect(rect(), Qt::black); // Activate to have a black background

  QRect bar = rect();

  bar.setTop(rect().top() + (1.0 - m_peakHoldLevel) * rect().height());
  bar.setBottom(bar.top() + 1);
  painter.fillRect(bar, m_rmsColor);
  bar.setBottom(rect().bottom());

  bar.setTop(rect().top() + (1.0 - m_decayedPeakLevel) * rect().height());
  painter.fillRect(bar, m_peakColor);

  bar.setTop(rect().top() + (1.0 - m_rmsLevel) * rect().height());
  painter.fillRect(bar, m_rmsColor);
}

AudioLevelMeter::AudioLevelMeter(QWidget *parent) : QWidget(parent), horizontalLayout(new QHBoxLayout(this)) {
  horizontalLayout->setSpacing(1);
  setToolTip(tr("Level meter to monitor the audio input"));
}

AudioLevelMeter::~AudioLevelMeter() { clear(); }

void AudioLevelMeter::addMeter(uint nbMeters, QWidget *parent) {
  for (uint i = 0; i < nbMeters; ++i) {
    levelMeters.push_back(new OneLevelMeter(parent));
    horizontalLayout->addWidget(levelMeters.last());
  }
}

void AudioLevelMeter::clear() {
  for (OneLevelMeter *meter : levelMeters) {
    delete meter;
  }
  levelMeters.clear();
}

void AudioLevelMeter::levelsChanged(const std::vector<double> &rmsLevels, const std::vector<double> &peakLevels) {
  if (rmsLevels.size() == peakLevels.size() && int(rmsLevels.size()) == levelMeters.size()) {
    int i = 0;
    for (OneLevelMeter *meter : levelMeters) {
      meter->levelChanged(rmsLevels[i], peakLevels[i]);
      i++;
    }
  } else {
    VideoStitch::Logger::error(tag) << "Wrong number of rms and peak values respectively (" << rmsLevels.size()
                                    << " and " << peakLevels.size() << ") expected " << levelMeters.size() << std::endl;
  }
}
