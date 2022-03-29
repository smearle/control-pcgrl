# -*- utf-8 -*-
#pylint: disable=W0603, E1101, I0011
'''Python implementation of sobol.cc

Please refer to:
http://web.maths.unsw.edu.au/%7Efkuo/sobol/index.html

And the following explanations:
http://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-notes.pdf

-----------------------------------------------------------------------------
Frances Y. Kuo

Email: <f.kuo@unsw.edu.au>
School of Mathematics and Statistics
University of New South Wales
Sydney NSW 2052, Australia

Last updated: 21 October 2008

You may incorporate this source code into your own program
  provided that you
  1) acknowledge the copyright owner in your program and publication
  2) notify the copyright owner by email
  3) offer feedback regarding your experience with different direction numbers

-----------------------------------------------------------------------------
Licence pertaining to sobol.cc and the accompanying sets of direction numbers
-----------------------------------------------------------------------------
Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the names of the copyright holders nor the names of the
      University of New South Wales and the University of Waikato
      and its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------------
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import numpy as np
from scipy.stats import norm
from ._directions import DIRECTIONS_D6_21201

__all__ = ['gen_sobol_vectors', 'gen_sobol_randn']


def gen_sobol_vectors(num, dim):
    """Generate sobol sequence, return ``num`` vectors for ``dim`` dimensions

    Notes
    -----
    Based on sobol.cc at:
    http://web.maths.unsw.edu.au/~fkuo/sobol/sobol.cc

    Parameters
    ----------
    num: int
        Number of vectors
    dim: int
        Number of dimensions

    Returns
    -------
    numpy.ndarray:
        A ``num`` * ``dim`` ``numpy`` matrix of generated Sobol vectors,
        each of size ``dim``
    """
    num = int(num)
    dim = int(dim)

    # v_l = max number of bits needed
    v_l = np.int64(np.ceil(np.log(num) / np.log(2.0)))
    inv2p32 = np.exp(-32.0 * np.log(2.0))

    # v_c[i] = index from the right of the first zero bit of i
    v_c = np.ndarray(num, dtype=np.int64)
    v_c[0] = 1
    for i in range(1, num):
        v_c[i] = 1
        value = i
        while value & 1:
            value >>= 1
            v_c[i] += 1

    # output[i][j] = the jth component of the ith point
    #                with i indexed from 0 to N-1 and j indexed from 0 to D-1
    output = np.ndarray((num, dim), dtype=np.float64)
    output.fill(0.0)

    # ----- Compute the first dimension -----

    # Compute direction numbers V[1] to V[L], scaled by pow(2,32)
    v_v = np.ndarray(num, dtype=np.int64)
    for i in range(0, v_l + 1):
        v_v[i] = 1 << (32 - i)

    # Evalulate X[0] to X[N-1], scaled by pow(2,32)
    v_x = np.ndarray(num, dtype=np.int64)
    v_x[0] = 0
    for i in range(1, num):
        v_x[i] = v_x[i - 1] ^ v_v[v_c[i - 1]]
        # Value for vector #i dimension #j==0
        output[i][0] = np.float64(v_x[i]) * inv2p32

    # ----- Compute the remaining dimensions -----
    for j in range(1, dim):
        # d_s is the degree of the primitive polynomial
        d_s = DIRECTIONS_D6_21201[j - 1][1]
        # d_a is the number representing the coefficient
        d_a = DIRECTIONS_D6_21201[j - 1][2]
        # d_m is the list of initial direction numbers
        d_m = [0] + DIRECTIONS_D6_21201[j - 1][3:]


        if v_l <= d_s:
            for i in range(1, v_l + 1):
                v_v[i] = d_m[i] << (32 - i)
        else:
            for i in range(1, d_s + 1):
                v_v[i] = d_m[i] << (32 - i)
            for i in range(d_s + 1, v_l + 1):
                v_v[i] = v_v[i - d_s] ^ (v_v[i - d_s] >> d_s)
                for k in range(1, d_s):
                    v_v[i] ^= (((d_a >> (d_s - 1 - k)) & 1) * v_v[i - k])

        v_x[0] = 0
        for i in range(1, num):
            v_x[i] = v_x[i - 1] ^ v_v[v_c[i - 1]]
            output[i][j] = np.float64(v_x[i]) * inv2p32
    # Skip first 0,...,0
    return output[1:]


def gen_sobol_randn(num):
    """Generate ``num`` quasi-random gaussian from a Sobol sequence

    Parameters
    ----------
    num: int
        Size of the numbers to generate

    Returns
    -------
    numpy.array:
        A vector of size ``num`` (mu=0, sigma=1) gaussians
    """
    return norm.ppf(gen_sobol_vectors(num, 1)[:, 0])
