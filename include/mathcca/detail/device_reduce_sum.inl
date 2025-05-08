#include <cstddef>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <mathcca/device_helper.h>
#include <mathcca/detail/device_shmem_proxy.h>
#include <mathcca/detail/nextPow2.h>

namespace cg = cooperative_groups;

namespace mathcca {
     
    template <Arithmetic T>
    __global__ void cg_reduce_kernel(const T* idata, T* odata, const std::size_t size, const T init) {
      // Shared memory for intermediate steps
      auto sdata = proxycca::shared_memory_proxy<T>();
      // Handle to thread block group
      cg::thread_block cta = cg::this_thread_block();
      // Handle to tile in thread block
      cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
      const auto ctaSize = static_cast<std::size_t>(cta.size());
      const auto numCtas = static_cast<std::size_t>(gridDim.x);
      const auto threadRank = static_cast<std::size_t>(cta.thread_rank());
      const auto threadIndex = static_cast<std::size_t>((blockIdx.x * ctaSize) + threadRank);
      auto threadVal{static_cast<T>(init)};
      {
        auto i{threadIndex};
        auto indexStride{numCtas * ctaSize};
        while (i < size) {
          threadVal += idata[i];
          i += indexStride;
        }
        sdata[threadRank] = threadVal;
      }
      // Wait for all tiles to finish and reduce within CTA
      {
        auto ctaSteps = static_cast<std::size_t>(tile.meta_group_size());
        auto ctaIndex = static_cast<std::size_t>(ctaSize >> 1);
        while (ctaIndex >= 32) {
          cta.sync();
          if (threadRank < ctaIndex) {
            threadVal += sdata[threadRank + ctaIndex];
            sdata[threadRank] = threadVal;
          }
          ctaSteps >>= 1;
          ctaIndex >>= 1;
        }
      }
      // Shuffle redux instead of smem redux
      {
        cta.sync();
        if (tile.meta_group_rank() == 0) {
          threadVal = cg::reduce(tile, threadVal, cg::plus<T>()); //cg_reduce_n(threadVal, tile);
        }
      }
      if (threadRank == 0)
        odata[blockIdx.x] = threadVal;
    }

    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    T reduce_sum(Cuda, const T* first, const T* last, const T init, cudaStream_t stream) {
      static_assert(THREAD_BLOCK_DIM <= 1024);
      std::size_t size{static_cast<std::size_t>(last - first)};
      constexpr unsigned int maxThreads{THREAD_BLOCK_DIM};
      unsigned int threads{size < (static_cast<std::size_t>(maxThreads) * 2) ? nextPow2((size + 1) / 2) : maxThreads};
      unsigned int blocks{static_cast<unsigned int>((size + static_cast<std::size_t>(threads * 2 - 1)) / static_cast<std::size_t>(threads * 2))};
      auto gpu_result{static_cast<T>(0)};
      T* d_odata = nullptr;
      T* d_intermediateSums = nullptr;
      checkCudaErrors(cudaMalloc((void **)&d_odata, blocks * sizeof(T)));
      checkCudaErrors(cudaMalloc((void **)&d_intermediateSums, sizeof(T) * blocks));
      dim3 dimBlock(threads, 1, 1);
      dim3 dimGrid(blocks, 1, 1);
      unsigned int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
      cg_reduce_kernel<T><<<dimGrid, dimBlock, smemSize, stream>>>(first, d_odata, size, init);
      // sum partial block sums on GPU
      unsigned int s{blocks};
      while (s > 1) {
        threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
        blocks  = (s + (threads * 2 - 1)) / (threads * 2);
        //std::cout << "s = " << s << "; threads = " << threads << "; blocks = " << blocks << "\n";
        checkCudaErrors(cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(T), cudaMemcpyDeviceToDevice));
        cg_reduce_kernel<T><<<blocks, threads, smemSize, stream>>>(d_intermediateSums, d_odata, s, static_cast<T>(0));
        s = (s + (threads * 2 - 1)) / (threads * 2);
      }
      checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaFree(d_odata));
      checkCudaErrors(cudaFree(d_intermediateSums));
      return gpu_result;
    }
     
}
	


