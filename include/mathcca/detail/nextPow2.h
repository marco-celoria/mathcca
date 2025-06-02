/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef NEXTPOW2_H
#define NEXTPOW2_H
    
namespace mathcca {
    
  namespace detail { 
    
    constexpr unsigned int nextPow2(unsigned int x) noexcept {
      x--;
      x |= x >> 1;
      x |= x >> 2;
      x |= x >> 4;
      x |= x >> 8;
      x |= x >> 16;
      return ++x;
    }
    
  } 
    
}    

#endif


