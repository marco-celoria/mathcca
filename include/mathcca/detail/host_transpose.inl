

namespace mathcca {


	      template<std::floating_point T>
    constexpr inline auto check_transposition_compatible_size(const host_matrix<T>& lhs, const host_matrix<T>& rhs) {
      if ((lhs.num_cols() == rhs.num_rows()) && (lhs.num_rows() == rhs.num_cols()))
        return true;
      return false;
    }



	    template<std::floating_point T>
  constexpr void transpose_parallel_Base(const host_matrix<T>& A, host_matrix<T>& B) {
    std::cout << "Base transposition\n";
    if (!check_transposition_compatible_size(A, B))
      throw std::length_error{"Incompatible sizes for matrix transposition"};
    using size_type= typename host_matrix<T>::size_type;
    const auto i_end{A.num_rows()};
    const auto j_end{A.num_cols()};
    #pragma omp parallel for collapse(2) default(shared)
    for (size_type i= 0; i < i_end; ++i) {
      for (size_type j= 0; j < j_end; ++j) {
        B(j, i)+= A(i, j);
      }
    }
  }

  template<std::floating_point T, unsigned int LINEAR_TILE_DIM>
  constexpr void transpose_parallel_Tiled(const host_matrix<T>& A, host_matrix<T>& B) {
    std::cout << "Tiled transposition\n";
    if (!check_transposition_compatible_size(A, B))
      throw std::length_error{"Incompatible sizes for matrix transposition"};
    using size_type= typename host_matrix<T>::size_type;
    const auto ii_end{A.num_rows()};
    const auto jj_end{A.num_cols()};
    const auto Ar_blocksize = std::min(static_cast<unsigned int>(ii_end), LINEAR_TILE_DIM);
    const auto Ac_blocksize = std::min(static_cast<unsigned int>(jj_end), LINEAR_TILE_DIM);
    #pragma omp parallel for collapse(2) default(shared)
    for (size_type ii= 0; ii < ii_end; ii+= Ar_blocksize) {
      for (size_type jj= 0; jj < jj_end; jj+= Ac_blocksize) {
        size_type i_end= std::min(ii + Ar_blocksize, ii_end);
        size_type j_end= std::min(jj + Ac_blocksize, jj_end);
        for (size_type i= ii; i < i_end; ++i) {
          for (size_type j= jj; j < j_end; ++j) {
            B(j, i)+= A(i, j);
          }
        }
      }
    }
  }

  #ifdef _MKL
  template<std::floating_point T>
  constexpr void transpose_parallel_Mkl(const host_matrix<T>& A, host_matrix<T>& B) {
    std::cout << "Mkl transposition\n";
    if (!check_transposition_compatible_size(A, B))
      throw std::length_error{"Incompatible sizes for matrix transposition"};
    const T alpha{static_cast<T>(1)};
    if constexpr(std::is_same_v<T, double>) {
      mkl_domatcopy ('R', 'T', A.num_rows(), A.num_cols(), alpha, A.data(), A.num_cols(), B.data(), B.num_cols());
    }
    else {
      mkl_somatcopy ('R', 'T', A.num_rows(), A.num_cols(), alpha, A.data(), A.num_cols(), B.data(), B.num_cols());
     }
  }
#endif

}
