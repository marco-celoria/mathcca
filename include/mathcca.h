/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifdef __CUDACC__
#include <mathcca/device_matrix.h>
#endif

#include <mathcca/host_matrix.h>
#include <mathcca/copy.h>
#include <mathcca/fill_const.h>
#include <mathcca/fill_iota.h>
#include <mathcca/fill_rand.h>
#include <mathcca/reduce_sum.h>
#include <mathcca/matmul.h>
#include <mathcca/transpose.h>
#include <mathcca/norm.h>
#include<iomanip>
#include<iostream>
#include<cmath>


