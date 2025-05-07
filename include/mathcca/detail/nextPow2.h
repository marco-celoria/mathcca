#ifndef NEXTPOW2_H
#define NEXTPOW2_H
  constexpr unsigned int nextPow2(unsigned int x) noexcept {
      x--;
      x |= x >> 1;
      x |= x >> 2;
      x |= x >> 4;
      x |= x >> 8;
      x |= x >> 16;
      return ++x;
    }
#endif
