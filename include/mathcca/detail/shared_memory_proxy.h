#ifndef SHARED_MEMORY_PROXY_H
#define SHARED_MEMORY_PROXY_H

namespace mathcca {
  // Utility class used to avoid linker errors with extern unsized shared memory arrays with templated type
  template <typename T>
  __device__ T* shared_memory_proxy() {
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T*>(memory);
  }
}

#endif
