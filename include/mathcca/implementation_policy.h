/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef IMPLEMENTATION_POLICY_H_
#define IMPLEMENTATION_POLICY_H_

namespace mathcca {

  namespace MM {
    class Base{};
    class Tiled{};
#ifdef _MKL
    class Mkl{};
#endif
#ifdef _CUBLAS
    class Cublas{};
#endif
  }

  namespace Trans {
    class Base{};
    class Tiled{};
#ifdef _MKL
    class Mkl{};
#endif
#ifdef _CUBLAS
    class Cublas{};
#endif
  }

  namespace Norm {
    class Base{};
#ifdef _MKL
    class Mkl{};
#endif
#ifdef _CUBLAS
    class Cublas{};
#endif
  }

}

#endif


