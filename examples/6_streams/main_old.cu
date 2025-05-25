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
  std::size_t N{l * m};
  std::size_t M{N/5};
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
  
  
   
  cudaStream_t stream0;
  cudaStreamCreate(&stream0);
  mathcca::copy(hA.cbegin(), hA.cend(), dA.begin(), stream0);
  mathcca::copy(hB.cbegin(), hB.cend(), dB.begin(), stream0);
  auto res_X= mathcca::reduce_sum(dA.begin(), dA.end(), static_cast<value_type>(0), stream0);
  dX= dA + dB;
  mathcca::copy(dX.cbegin(), dX.cend(), hX.begin(), stream0);
  cudaDeviceSynchronize();
  std::cout << "res_X= " << res_X << "\n";
  cudaStreamDestroy(stream0);
 
  cudaStream_t stream1, stream2;
  
  cudaStreamCreate( &stream1 );
  cudaStreamCreate( &stream2 );

  value_type res_1= static_cast<value_type>(0);
  value_type res_2= static_cast<value_type>(0);
  value_type res_Y= static_cast<value_type>(0);
  for (auto i= 0; i< 5; ++i) {
    mathcca::detail::copy(mathcca::CudaHtoDcpy(), hA.cbegin().get() + static_cast<std::ptrdiff_t>(i*M),      hA.cbegin().get() + static_cast<std::ptrdiff_t>(i*M+M/2), dA_1.begin().get(), stream1);
    mathcca::detail::copy(mathcca::CudaHtoDcpy(), hA.cbegin().get() + static_cast<std::ptrdiff_t>(i*M+M/2) , hA.cbegin().get() + static_cast<std::ptrdiff_t>(i*M+M),   dA_2.begin().get(), stream2);
    mathcca::detail::copy(mathcca::CudaHtoDcpy(), hB.cbegin().get() + static_cast<std::ptrdiff_t>(i*M),      hB.cbegin().get() + static_cast<std::ptrdiff_t>(i*M+M/2), dB_1.begin().get(), stream1);
    mathcca::detail::copy(mathcca::CudaHtoDcpy(), hB.cbegin().get() + static_cast<std::ptrdiff_t>(i*M+M/2) , hB.cbegin().get() + static_cast<std::ptrdiff_t>(i*M+M),   dB_2.begin().get(), stream2);
    res_1= mathcca::detail::reduce_sum(mathcca::Cuda(), dA_1.cbegin().get(), dA_1.cbegin().get() + static_cast<std::ptrdiff_t>(M/2), static_cast<value_type>(0), stream1);
    res_2= mathcca::detail::reduce_sum(mathcca::Cuda(), dA_2.cbegin().get(), dA_2.cbegin().get() + static_cast<std::ptrdiff_t>(M/2), static_cast<value_type>(0), stream2);
    res_Y+= res_1 + res_2;
    dY_1= dA_1 + dB_1;
    dY_2= dA_2 + dB_2;
    mathcca::host_iterator<value_type, false> h_it_1{hY.begin().get() + static_cast<std::ptrdiff_t>(i*M)};
    mathcca::host_iterator<value_type, false> h_it_2{hY.begin().get() + static_cast<std::ptrdiff_t>(i*M + M/2)};
    mathcca::copy(dY_1.cbegin(), dY_1.cend(), h_it_1, stream1);
    mathcca::copy(dY_2.cbegin(), dY_2.cend(), h_it_2, stream2);
  }
  cudaDeviceSynchronize();
  std::cout << "res_Y= " << res_Y << "\n";
  std::cout << "res_Y - res_X= " << res_Y - res_X << "\n";
  std::cout << "hX == hY? " << std::boolalpha << (hX == hY) << std::noboolalpha << "\n";
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
}


