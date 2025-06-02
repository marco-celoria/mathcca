/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef ARITHMETIC_TYPE_H_
#define ARITHMETIC_TYPE_H_

#include <type_traits>

namespace mathcca {
    
  template<typename T>
  concept Arithmetic = std::is_arithmetic_v<T>;
     
}

#endif

