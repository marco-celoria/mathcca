/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef SHARED_MEMORY_PROXY_H
#define SHARED_MEMORY_PROXY_H

namespace mathcca {

  namespace detail {
    
    // Utility class used to avoid linker errors with extern unsized shared memory arrays with templated type
    template <typename T>
    __device__ T* shared_memory_proxy() {
      extern __shared__ unsigned char memory[];
      return reinterpret_cast<T*>(memory);
    }

  }

}

#endif


