#include <mathcca/host_matrix.h>
#include <mathcca/device_matrix.h>
#include <mathcca/norm.h>
#include <mathcca/detail/norm_impl.h>
#include <iostream>
#include <mathcca/fill_rand.h>
#include <mathcca/fill_const.h>
#include <mathcca/detail/copy_impl.h>
#include <mathcca/detail/reduce_sum_impl.h>
#include <mathcca/reduce_sum.h>
#include <mathcca/copy.h>
#include <mathcca/host_iterator.h>
int main(int argc, char **argv)  {
  std::size_t l{1'000};
  std::size_t m{1'000'000};
  std::size_t nElem{l * m};
  constexpr int NSTREAM= 10;
  std::size_t iElem{nElem/NSTREAM};
#ifdef _USE_DOUBLE_PRECISION
  using value_type= double;
#else
  using value_type= float;
#endif
  mathcca::host_matrix<value_type>   hA{l, m};
  mathcca::host_matrix<value_type>   hB{l, m};
  mathcca::host_matrix<value_type>   hX{l, m};
  mathcca::host_matrix<value_type>   hY{l, m};

  mathcca::device_matrix<value_type> dA{l, m};
  mathcca::device_matrix<value_type> dB{l, m};
  mathcca::device_matrix<value_type> dX{l, m};

  mathcca::device_matrix<value_type> dA_1{l/10, m};
  mathcca::device_matrix<value_type> dA_2{l/10, m};
  mathcca::device_matrix<value_type> dB_1{l/10, m};
  mathcca::device_matrix<value_type> dB_2{l/10, m};
  mathcca::device_matrix<value_type> dY_1{l/10, m};
  mathcca::device_matrix<value_type> dY_2{l/10, m};

  mathcca::fill_rand(hA.begin(), hA.end());
 
  cudaEvent_t startX; cudaEventCreate(&startX);
  cudaEvent_t stopX; cudaEventCreate(&stopX);
  cudaEvent_t startY1; cudaEventCreate(&startY1);
  cudaEvent_t startY2; cudaEventCreate(&startY2);
  cudaEvent_t stopY1; cudaEventCreate(&stopY1);
  cudaEvent_t stopY2; cudaEventCreate(&stopY2);

  value_type res_X= static_cast<value_type>(0);
  cudaStream_t stream0;
  cudaStreamCreate(&stream0);
  cudaEventRecord(startX,stream0);
  mathcca::copy(hA.cbegin(), hA.cend(), dA.begin(), stream0);
  mathcca::copy(hB.cbegin(), hB.cend(), dB.begin(), stream0);
  res_X= mathcca::reduce_sum(dA.begin(), dA.end(), static_cast<value_type>(0), stream0);
  dX= (dA + dB)*static_cast<value_type>(2);
  mathcca::copy(dX.cbegin(), dX.cend(), hX.begin(), stream0);
  cudaDeviceSynchronize();
  cudaEventRecord(stopX,stream0);
  std::cout << "res_X= " << res_X << "\n";
  cudaStreamDestroy(stream0);
 
  cudaStream_t stream1, stream2;
  
  cudaStreamCreate( &stream1 );
  cudaStreamCreate( &stream2 );

  value_type res_1= static_cast<value_type>(0);
  value_type res_2= static_cast<value_type>(0);
  value_type res_Y= static_cast<value_type>(0);
  cudaEventRecord(startY1,stream1);
  cudaEventRecord(startY2,stream2);
  for (auto i= 0; i< nElem; i+= 2*iElem) {
    mathcca::detail::copy(mathcca::CudaHtoDcpy(), hA.cbegin().get() + static_cast<std::ptrdiff_t>(i),      hA.cbegin().get() + static_cast<std::ptrdiff_t>(i+iElem), dA_1.begin().get(), stream1);
    mathcca::detail::copy(mathcca::CudaHtoDcpy(), hA.cbegin().get() + static_cast<std::ptrdiff_t>(i+iElem) , hA.cbegin().get() + static_cast<std::ptrdiff_t>(i+2*iElem), dA_2.begin().get(), stream2);
    mathcca::detail::copy(mathcca::CudaHtoDcpy(), hB.cbegin().get() + static_cast<std::ptrdiff_t>(i),      hB.cbegin().get() + static_cast<std::ptrdiff_t>(i+iElem), dB_1.begin().get(), stream1);
    mathcca::detail::copy(mathcca::CudaHtoDcpy(), hB.cbegin().get() + static_cast<std::ptrdiff_t>(i+iElem) , hB.cbegin().get() + static_cast<std::ptrdiff_t>(i+2*iElem), dB_2.begin().get(), stream2);
    res_1= mathcca::detail::reduce_sum(mathcca::Cuda(), dA_1.cbegin().get(), dA_1.cbegin().get() + static_cast<std::ptrdiff_t>(iElem), static_cast<value_type>(0), stream1);
    res_2= mathcca::detail::reduce_sum(mathcca::Cuda(), dA_2.cbegin().get(), dA_2.cbegin().get() + static_cast<std::ptrdiff_t>(iElem), static_cast<value_type>(0), stream2);
    res_Y+= res_1 + res_2;
    dY_1= (dA_1 + dB_1)*static_cast<value_type>(2);
    dY_2= (dA_2 + dB_2)*static_cast<value_type>(2);
    mathcca::host_iterator<value_type, false> h_it_1{hY.begin().get() + static_cast<std::ptrdiff_t>(i)};
    mathcca::host_iterator<value_type, false> h_it_2{hY.begin().get() + static_cast<std::ptrdiff_t>(i+iElem)};
    mathcca::copy(dY_1.cbegin(), dY_1.cend(), h_it_1, stream1);
    mathcca::copy(dY_2.cbegin(), dY_2.cend(), h_it_2, stream2);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(stopY1,stream1);
  cudaEventRecord(stopY2,stream2);
  std::cout << "res_Y= " << res_Y << "\n";
  std::cout << "res_Y - res_X= " << res_Y - res_X << "\n";
  std::cout << "hX == hY? " << std::boolalpha << (hX == hY) << std::noboolalpha << "\n";
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  float time_X = 0.0f;
  float time_Y1 = 0.0f;
  float time_Y2 = 0.0f;
  cudaEventElapsedTime(&time_X,  startX,  stopX);
  cudaEventElapsedTime(&time_Y1, startY1, stopY1);
  cudaEventElapsedTime(&time_Y2, startY2, stopY2);
  cudaEventDestroy(startX);
  cudaEventDestroy(startY1);
  cudaEventDestroy(startY2);
  cudaEventDestroy(stopX);
  cudaEventDestroy(stopY1);
  cudaEventDestroy(stopY2);
  std::cout << time_X << " " << time_Y1 << " " << time_Y2 << "\n";
}


