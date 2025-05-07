

namespace mathcca {

  namespace matricca {
 
    template<std::floating_point T>
    constexpr inline auto check_matmul_compatible_size(const host_matrix<T>& lhs, const host_matrix<T>& rhs) {
      if (lhs.num_cols() == rhs.num_rows())
        return true;
      return false;
    }

        template<std::floating_point T>
    constexpr void mm_parallel_Base(const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C) {
      std::cout << "Base matmul\n";
      if (!check_matmul_compatible_size(A, B))
        throw std::length_error{"Incompatible length matrix-matrix product"};
      using size_type= typename host_matrix<T>::size_type;
      using value_type= T;
      const auto i_end{A.num_rows()};
      const auto j_end{B.num_cols()};
      const auto k_end{A.num_cols()};
      #pragma omp parallel for collapse(2) default(shared)
      for (size_type i= 0; i < i_end; ++i) {
        for (size_type j= 0; j < j_end; ++j) {
          auto sum{static_cast<value_type>(0)};
          #pragma omp simd reduction(+:sum)
          for (size_type k= 0; k < k_end; ++k) {
            sum+= A(i, k) * B(k, j);
          }
          C(i, j)= sum;
        }
      }
    }

	  template<std::floating_point T, unsigned int LINEAR_TILE_DIM>
  constexpr void mm_parallel_Tiled(const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C) {
    std::cout << "Tiled matmul\n";
    if (!check_matmul_compatible_size(A, B))
      throw std::length_error{"Incompatible length matrix-matrix product"};
    using size_type= typename host_matrix<T>::size_type;
    const auto ii_end{A.num_rows()};
    const auto jj_end{B.num_cols()};
    const auto kk_end{A.num_cols()};
    const auto Ar_blocksize = std::min(static_cast<unsigned int>(ii_end), LINEAR_TILE_DIM);
    const auto Bc_blocksize = std::min(static_cast<unsigned int>(jj_end), LINEAR_TILE_DIM);
    const auto Ac_blocksize = std::min(static_cast<unsigned int>(kk_end), LINEAR_TILE_DIM);
    #pragma omp parallel for collapse(2) default(shared)
    for (size_type ii= 0; ii < ii_end; ii+= Ar_blocksize) {
      for (size_type jj= 0; jj < jj_end; jj+= Bc_blocksize) {
        size_type i_end= std::min(ii + Ar_blocksize, ii_end);
        size_type j_end= std::min(jj + Bc_blocksize, jj_end);
        for (size_type kk= 0; kk < kk_end; kk+= Ac_blocksize) {
          size_type k_end = std::min(kk + Ac_blocksize, kk_end);
          for (size_type i= ii; i < i_end; ++i) {
            for (size_type k= kk; k < k_end; ++k) {
              for (size_type j= jj; j < j_end; ++j) {
                C(i, j)+= A(i, k) * B(k, j);
              }
            }
          }
        }
      }
    }
  }

	    #ifdef _MKL
  template<std::floating_point T>
  constexpr void mm_parallel_Mkl(const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C) {
    std::cout << "Mkl matmul\n";
    if (!check_matmul_compatible_size(A, B))
      throw std::length_error{"Incompatible length matrix-matrix product"};
    //using size_type= typename host_matrix<T>::size_type;
    using value_type= T;
    value_type alpha{1};
    value_type beta{0};
    if constexpr(std::is_same_v<T,double>) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.num_rows(), B.num_cols(), A.num_cols(),
               alpha, A.data(), A.num_cols(), B.data(), B.num_cols(), beta, C.data(), C.num_cols());
    } else {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.num_rows(), B.num_cols(), A.num_cols(),
               alpha, A.data(), A.num_cols(), B.data(), B.num_cols(), beta, C.data(), C.num_cols());
    }
  }
  #endif
 
  }
} 
