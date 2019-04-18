// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/matrix.hpp"

#include <iostream>

namespace VideoStitch {

template <typename T>
void Vector3<T>::print(std::ostream& s) const {
  s << "[" << v[0] << "," << v[1] << "," << v[2] << "]";
}

template <typename T>
std::ostream& operator<<(std::ostream& s, const Vector3<T>& v) {
  v.print(s);
  return s;
}

template std::ostream& operator<<<double>(std::ostream& s, const Vector3<double>& m);

template <typename T>
void Matrix33<T>::print(std::ostream& s) const {
  s << std::endl;
  s << "[" << m[0][0] << " " << m[0][1] << " " << m[0][2] << "]" << std::endl;
  s << "[" << m[1][0] << " " << m[1][1] << " " << m[1][2] << "]" << std::endl;
  s << "[" << m[2][0] << " " << m[2][1] << " " << m[2][2] << "]" << std::endl;
}

template <typename T>
std::ostream& operator<<(std::ostream& s, const Matrix33<T>& m) {
  m.print(s);
  return s;
}

template std::ostream& operator<<<double>(std::ostream& s, const Matrix33<double>& m);

// explicit instantiations
template class Matrix33<double>;
template class Vector3<double>;
}  // namespace VideoStitch
