// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/circularBuffer.hpp"

namespace VideoStitch {
namespace Testing {

void basicCircularBufferTest() {
  int in[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  // Construction

  CircularBuffer<int> cb1(16);
  ENSURE(cb1.capacity() == 16);

  // Check write & read without passing the end

  size_t size = 8;
  cb1.push(in, 8);  // 0, 1, 2, 3, 4, 5, 6, 7
  ENSURE(cb1.size() == size);

  cb1.push(in, 5);  // 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4
  size += 5;
  ENSURE(cb1.size() == size);
  ENSURE(cb1[0] == 0);
  ENSURE(cb1[9] == 1);

  int out1[4];
  cb1.pop(out1, 4);  // 0, 1, 2, 3 <-- 4, 5, 6, 7, 0, 1, 2, 3, 4
  size -= 4;
  ENSURE(cb1.size() == size);
  ENSURE(out1[0] == 0);
  ENSURE(out1[3] == 3);
  ENSURE(cb1[0] == 4);
  ENSURE(cb1[6] == 2);

  // Check write & read passing the end

  cb1.push(in, 5);  // 4, 5, 6, 7, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4
  size += 5;
  ENSURE(cb1.size() == size);

  int out2[9];
  size -= 9;
  cb1.pop(out2, 9);  // 4, 5, 6, 7, 0, 1, 2, 3, 4 <-- 0, 1, 2, 3, 4
  ENSURE(cb1.size() == size);
  ENSURE(out2[0] == 4);
  ENSURE(out2[0] == 4);

  size += 16;
  cb1.push(in, 8);  // 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7
  cb1.push(in, 8);  // 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7
  ENSURE(cb1.capacity() > 8);
  ENSURE(cb1.size() == size);

  int* out3 = new int[size];
  cb1.pop(out3, size);
  ENSURE(out3[0] == 0);
  ENSURE(out3[5] == 0);
  ENSURE(out3[7] == 2);
  ENSURE(out3[14] == 1);
  delete[] out3;
  ENSURE(cb1.size() == 0);

  // Check write & read passing when reaching the end exactly

  CircularBuffer<int> cb2(8);
  ENSURE(cb2.capacity() == 8);
  cb2.push(in, 8);  // 0, 1, 2, 3, 4, 5, 6, 7
  cb2[1] = 1;
  ENSURE(cb2[1] == 1);
  cb2.erase(4);  // 4, 5, 6, 7
  ENSURE(cb2[0] == 4);
  ENSURE(cb2[2] == 6);
  int out4[4];
  cb2.push(in, 4);   // 4, 5, 6, 7, 0, 1, 2, 3
  cb2.pop(out4, 4);  // 4, 5, 6, 7 <-- 0, 1, 2, 3
  ENSURE(out4[0] == 4);
  ENSURE(cb2[0] == 0);
  ENSURE(cb2[2] == 2);
  size_t r = cb2.pop(out4, 10);  // 0, 1, 2, 3 <--
  ENSURE(r == 4);
  ENSURE(out4[0] == 0);

  // Check push items and [] operator

  CircularBuffer<int> cb3(8);
  ENSURE(cb3.capacity() == 8);
  cb3.push(0);
  cb3.push(0);
  cb3.erase(2);
  for (int i = 0; i != 8; ++i) {
    cb3.push(i);
    ENSURE(cb3[i] == i);
  }
  int out5[8];
  cb3.pop(out5, 8);
  ENSURE(out5[0] == 0);
  ENSURE(out5[7] == 7);
  for (int i = 0; i != 16; ++i) {
    cb3.push(i);
    ENSURE(cb3[0] == i);
    cb3.erase(1);
  }

  // Check assign

  CircularBuffer<int> cb4;
  cb4.assign(128, -1);
  ENSURE(cb4.size() == 128);
  ENSURE(cb4.capacity() >= 128);
  ENSURE(cb4[7] == -1);

  CircularBuffer<int> cb6;
  ENSURE(cb6.size() == 0);
  cb6.assign(1, 1);
  ENSURE(cb6[0] == 1);
  ENSURE(cb6.size() == 1);
  cb6.push(2);
  ENSURE(cb6[1] == 2);
  ENSURE(cb6.size() == 2);

  // Check when size == 1

  CircularBuffer<int> cb5(1);
  cb5.push(1);
  ENSURE(cb5[0] == 1);
  ENSURE(cb5.size() == 1);
  cb5.push(2);
  ENSURE(cb5[1] == 2);
  ENSURE(cb5.size() == 2);
  ENSURE(cb5.size() <= cb5.capacity());
  ENSURE(cb5[0] == 1);

  // Check when size == 0

  CircularBuffer<int> cb7;
  cb7.push(1);
  ENSURE(cb7[0] == 1);
  ENSURE(cb7.size() == 1);
  cb7.push(2);
  ENSURE(cb7[1] == 2);
  ENSURE(cb7.size() == 2);

  // Test slicing
  {
    CircularBuffer<int> cb8(8);
    cb8.assign(4, -1);

    CircularBuffer<int>::Slice s0 = cb8.slice(0);
    ENSURE(s0.first.size() == 0 && s0.second.size() == 0);
    CircularBuffer<int>::Slice s1 = cb8.slice(1);
    ENSURE(s1.first.size() == 1 && s1.second.size() == 0);
    ENSURE(s1.first.begin() == &cb8[0]);
    CircularBuffer<int>::Slice s2 = cb8.slice(4);
    ENSURE(s2.first.size() == 4 && s2.second.size() == 0);
    CircularBuffer<int>::Slice s3 = cb8.slice(8);
    ENSURE(s3.first.size() == 4 && s3.second.size() == 0);

    int offset = 2;
    CircularBuffer<int>::Slice s4 = cb8.slice(2, offset);
    ENSURE(s4.first.size() == 2 && s4.second.size() == 0);
    ENSURE(s4.first.begin() == &cb8[2]);
    CircularBuffer<int>::Slice s5 = cb8.slice(4, offset);
    ENSURE(s5.first.size() == 2 && s5.second.size() == 0);
  }

  {
    CircularBuffer<int> cb8(8);
    cb8.assign(8, -1);
    cb8.erase(4);
    cb8.push(-2);
    cb8.push(-2);  // 4 + 2 pass the end

    CircularBuffer<int>::Slice s0 = cb8.slice(0);
    ENSURE(s0.first.size() == 0 && s0.second.size() == 0);
    CircularBuffer<int>::Slice s1 = cb8.slice(1);
    ENSURE(s1.first.size() == 1 && s1.second.size() == 0);
    ENSURE(s1.first.begin() == &cb8[0]);
    CircularBuffer<int>::Slice s2 = cb8.slice(6);
    ENSURE(s2.first.size() == 4 && s2.second.size() == 2);
    CircularBuffer<int>::Slice s3 = cb8.slice(8);
    ENSURE(s3.first.size() == 4 && s3.second.size() == 2);

    int offset = 2;
    CircularBuffer<int>::Slice s4 = cb8.slice(2, offset);
    ENSURE(s4.first.size() == 2 && s4.second.size() == 0);
    ENSURE(s4.first.begin() == &cb8[offset]);
    CircularBuffer<int>::Slice s5 = cb8.slice(4, offset);
    ENSURE(s5.first.size() == 2 && s5.second.size() == 2);
    ENSURE(s5.first.begin() == &cb8[offset]);
    ENSURE(s5.second.end() == &cb8[offset + 4]);
    CircularBuffer<int>::Slice s6 = cb8.slice(6, offset);
    ENSURE(s6.first.size() == 2 && s6.second.size() == 2);
    ENSURE(s6.first.begin() == &cb8[offset]);
    ENSURE(s6.second.end() == &cb8[offset + 4]);

    offset = 4;
    CircularBuffer<int>::Slice s7 = cb8.slice(2, offset);
    ENSURE(s7.first.size() == 2 && s7.second.size() == 0);
    ENSURE(s7.first.begin() == &cb8[offset]);
    ENSURE(s7.first.end() == &cb8[offset + 2]);
  }
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::basicCircularBufferTest();

  return 0;
}
