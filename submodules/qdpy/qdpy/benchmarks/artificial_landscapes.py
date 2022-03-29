#    This file is part of qdpy.
#
#    qdpy is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    qdpy is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with qdpy. If not, see <http://www.gnu.org/licenses/>.

"""Collection of test functions and benchmarks for QD algorithms based on artificial landscapes. Cf https://arxiv.org/pdf/1308.4008v1.pdf and https://arxiv.org/pdf/1207.4318.pdf and https://en.wikipedia.org/wiki/Test_functions_for_optimization and https://www.sfu.ca/~ssurjano/optimization.html"""

__all__ = [
        "sphere", "illumination_sphere", "SphereBenchmark",
        "weighted_sphere", "illumination_weighted_sphere", "WeightedSphereBenchmark",
        "rotated_hyper_ellipsoid", "illumination_rotated_hyper_ellipsoid", "RotatedHyperEllipsoidBenchmark",
        "rosenbrock", "illumination_rosenbrock", "RosenbrockBenchmark",
        "rastrigin_normalised", "illumination_rastrigin_normalised", "NormalisedRastriginBenchmark",
        "rastrigin", "illumination_rastrigin", "RastriginBenchmark",
        "schwefel", "illumination_schwefel", "SchwefelBenchmark",
        "small_schwefel", "illumination_small_schwefel", "SmallSchwefelBenchmark",
        "griewangk", "illumination_griewangk", "GriewangkBenchmark",
        "sum_of_powers", "illumination_sum_of_powers", "SumOfPowersBenchmark",
        "ackley", "illumination_ackley", "AckleyBenchmark",
        "styblinski_tang", "illumination_styblinski_tang", "StyblinskiTangBenchmark",
        "levy", "illumination_levy", "LevyBenchmark",
        "perm0db", "illumination_perm0db", "Perm0dbBenchmark",
        "permdb", "illumination_permdb", "PermdbBenchmark",
        "trid", "illumination_trid", "TridBenchmark",
        "zakharov", "illumination_zakharov", "ZakharovBenchmark",
        "dixon_price", "illumination_dixon_price", "DixonPriceBenchmark",
        "powell", "illumination_powell", "PowellBenchmark",
        "michalewicz", "illumination_michalewicz", "MichalewiczBenchmark",
        "wavy", "illumination_wavy", "WavyBenchmark",
        "trigonometric02", "illumination_trigonometric02", "Trigonometric02Benchmark",
        "qing", "illumination_qing", "QingBenchmark",
        "small_qing", "illumination_small_qing", "SmallQingBenchmark",
        "deb01", "illumination_deb01", "Deb01Benchmark",
        "shubert04", "illumination_shubert04", "Shubert04Benchmark",
        ]

from math import cos, pi
from typing import Optional, Tuple, List, Iterable, Iterator, Any, TypeVar, Generic, Union, Sequence, MutableSet, MutableSequence, Type, Callable, Generator, Mapping, MutableMapping, overload
import numpy as np

from qdpy import algorithms
from qdpy.base import *
from .base import *


def _illuminate_artificial_landscape(ind, func, nb_features = 2):
    fitness = func(ind)
    features = list(ind[:nb_features])
    return fitness, features


def sphere(ind):
    """Sphere function defined as:
    $$ f(x) = \sum_{i=1}^{n} x_i^{2} $$
    with a search domain of $-5.12 < x_i < 5.12, 1 \leq i \leq n$.
    The global minimum is at $f(x_1, ..., x_n) = f(0, ..., 0) = 0.
    It is continuous, convex and unimodal.
    """
    return sum((x ** 2. for x in ind)),
illumination_sphere = algorithms.partial(_illuminate_artificial_landscape, func=sphere) # type: ignore
class SphereBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-5.12, 5.12)
        super().__init__(fn=algorithms.partial(illumination_sphere, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")

def weighted_sphere(ind):
    """Weighted sphere function defined as:
    $$ f(x) = \sum_{i=1}^{n} i \times x_i^{2} $$
    with a search domain of $-5.12 < x_i < 5.12, 1 \leq i \leq n$.
    The global minimum is at $f(x_1, ..., x_n) = f(0, ..., 0) = 0.
    It is continuous, convex and unimodal.
    """
    return sum((i * (ind[i] ** 2.) for i in range(len(ind)))),
illumination_weighted_sphere = algorithms.partial(_illuminate_artificial_landscape, func=weighted_sphere) # type: ignore
class WeightedSphereBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-5.12, 5.12)
        super().__init__(fn=algorithms.partial(illumination_weighted_sphere, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")


def rotated_hyper_ellipsoid(ind):
    """Rotated hyper-ellipsoid function defined as:
    $$ f(x) = \sum_{i=1}^{n} \sum_{j=1}^{i} x_j^{2} $$
    with a search domain of $-65.536 < x_i < 65.536, 1 \leq i \leq n$.
    The global minimum is at $f(x_1, ..., x_n) = f(0, ..., 0) = 0.
    It is continuous, convex and unimodal.
    """
    return sum(( sum((ind[j] ** 2. for j in range(i))) for i in range(len(ind)) )),
illumination_rotated_hyper_ellipsoid = algorithms.partial(_illuminate_artificial_landscape, func=rotated_hyper_ellipsoid) # type: ignore
class RotatedHyperEllipsoidBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-65.536, 65.536)
        super().__init__(fn=algorithms.partial(illumination_rotated_hyper_ellipsoid, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")


def rosenbrock(ind):
    """Rosenbrock function defined as:
    $$ f(x) = \sum_{i=1}^{n-1} 100 \times (x_{i+1} - x_i^{2})^{2} + (1 - x_{i})^{2} $$
    with a search domain of $-2.048 < x_i < 2.048, 1 \leq i \leq n$.
    The global minimum is at $f(x_1, ..., x_n) = f(1, ..., 1) = 0.
    """
    return sum((100. * (ind[i+1]-ind[i]**2.)**2. + (1-ind[i])**2. for i in range(len(ind)-1))),
illumination_rosenbrock = algorithms.partial(_illuminate_artificial_landscape, func=rosenbrock) # type: ignore
class RosenbrockBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-2.048, 2.048)
        super().__init__(fn=algorithms.partial(illumination_rosenbrock, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")



def rastrigin_normalised(ind, a=10.):
    """Rastrigin test function, with inputs and outputs scaled to be between 0. and 1."""
    n = float(len(ind))
    ind_n = [-5.12 + x * (5.12 * 2.) for x in ind]
    return (a * n + sum(x * x - a * cos(2. * pi * x) for x in ind_n)) / (a * n + n * (5.12 * 5.12 + a)),
illumination_rastrigin_normalised = algorithms.partial(_illuminate_artificial_landscape, func=rastrigin_normalised) # type: ignore
class NormalisedRastriginBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (0., 1.)
        super().__init__(fn=algorithms.partial(illumination_rastrigin_normalised, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., 1.),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")


def rastrigin(ind, a=10.):
    """Rastrigin test function."""
    n = float(len(ind))
    return a * n + sum(x * x - a * cos(2. * pi * x) for x in ind),
illumination_rastrigin = algorithms.partial(_illuminate_artificial_landscape, func=rastrigin) # type: ignore
class RastriginBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-5.12, 5.12)
        super().__init__(fn=algorithms.partial(illumination_rastrigin, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")



def schwefel(ind):
    """Schwefel function defined as:
    $$ f(x) = 418.9829n - \sum_{i=1}^{n} x_i \sin(\sqrt(\norm(x_i))) $
    with a search domain of $-500 < x_i < 500, 1 \leq i \leq n$.
    The global minimum is at $f(x_1, ..., x_n) = f(420.9687, ..., 420.9687) = 0.
    """
    return 418.9829 * float(len(ind)) - sum((x * math.sin(math.sqrt(abs(x))) for x in ind)),
illumination_schwefel = algorithms.partial(_illuminate_artificial_landscape, func=schwefel) # type: ignore
class SchwefelBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-500., 500.)
        super().__init__(fn=algorithms.partial(illumination_schwefel, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")



def small_schwefel(ind):
    """Schwefel function defined as:
    $$ f(x) = 418.9829n - \sum_{i=1}^{n} x_i \sin(\sqrt(\norm(x_i))) $
    with a search domain of $-100 < x_i < 100, 1 \leq i \leq n$.
    """
    return 418.9829 * float(len(ind)) - sum((x * math.sin(math.sqrt(abs(x))) for x in ind)),
illumination_small_schwefel = algorithms.partial(_illuminate_artificial_landscape, func=small_schwefel) # type: ignore
class SmallSchwefelBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        #ind_domain = (-50., 50.)
        ind_domain = (0., 500.)
        super().__init__(fn=algorithms.partial(illumination_small_schwefel, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")


def griewangk(ind):
    """Griewangk function defined as:
    $$ f(x) = 1 + \sum_{i=1}^{n} \frac{x_i^2}{4000} - \prod_{i=1}^n \cos \frac{x_i}{\sqrt{i}} $$
    with a search domain of $-600 < x_i < 600, 1 \leq i \leq n$.
    The global minimum is at $f(x_1, ..., x_n) = f(0, ..., 0) = 0.
    """
    return 1. + sum(( (x ** 2.) / 4000. for x in ind)) - np.prod(list((math.cos(ind[i] / math.sqrt(i+1.)) for i in range(len(ind))))),
illumination_griewangk = algorithms.partial(_illuminate_artificial_landscape, func=griewangk) # type: ignore
class GriewangkBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-600., 600.)
        super().__init__(fn=algorithms.partial(illumination_griewangk, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")



def sum_of_powers(ind):
    """Sum Of Powers function defined as:
    $$ f(x) = \sum_{i=1}^{n} \norm{x_i}^{(i+1)} $$
    with a search domain of $-1 < x_i < 1, 1 \leq i \leq n$.
    The global minimum is at $f(x_1, ..., x_n) = f(0, ..., 0) = 0.
    """
    return sum((abs(ind[i]) ** (i+1.) for i in range(len(ind)))),
illumination_sum_of_powers = algorithms.partial(_illuminate_artificial_landscape, func=sum_of_powers) # type: ignore
class SumOfPowersBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-1., 1.)
        super().__init__(fn=algorithms.partial(illumination_sum_of_powers, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")



def ackley(ind, a=20., b=0.2, c=2.*math.pi):
    """Ackley function defined as:
    $$ f(x) = -a \exp^{-b \sqrt{1/n \sum_{i=1}^n x_i^2} - \exp^{1/n \sum_{i=1}^n \cos{c x_i}} + a + \exp^{1} $$
    with $a=20, b=0.2, c=2\pi$ and with a search domain of $-32.768 < x_i < 32.768, 1 \leq i \leq n$.
    The global minimum is at $f(x_1, ..., x_n) = f(0, ..., 0) = 0.
    """
    return - a * math.exp(-b * math.sqrt(1./len(ind) * sum((x**2. for x in ind)) )) - math.exp(1./len(ind) * sum((math.cos(c*ind[i]) for i in range(len(ind)) )) ) + a + math.exp(1.),
illumination_ackley = algorithms.partial(_illuminate_artificial_landscape, func=ackley) # type: ignore
class AckleyBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-32.768, 32.768)
        super().__init__(fn=algorithms.partial(illumination_ackley, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")



def styblinski_tang(ind):
    """Styblinski-Tang function defined as:
    $$ f(x) = 1/2 \sum_{i=1}^{n} x_i^4 - 16 x_i^2 + 5 x_i $$
    with a search domain of $-5 < x_i < 5, 1 \leq i \leq n$.
    """
    return sum(((x ** 4.) - 16. * (x ** 2.) + 5. * x for x in ind)) / 2.,
illumination_styblinski_tang = algorithms.partial(_illuminate_artificial_landscape, func=styblinski_tang) # type: ignore
class StyblinskiTangBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-5, 5)
        super().__init__(fn=algorithms.partial(illumination_styblinski_tang, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((-math.inf, math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")



def levy(ind):
    """Levy function defined as:
    $$ f(x) = \sin(\pi w_1)^2 + \sum_{i=1}^{n-1}(w_i-1)^2 (1+10 \sin(\pi w_i+1)^2) + (w_n-1)^2 (1+\sin(2\pi w_d)^2)$$
    where $w_i = 1+ \frac{x_i-1}{4}$
    with a search domain of $-10 < x_i < 10, 1 \leq i \leq n$.
    The global minimum is at $f(x_1, ..., x_n) = f(1, ..., 1) = 0.
    """
    w = list((1. + (x-1.)/4. for x in ind))
    return (math.sin(math.pi * w[0]))**2. + \
        sum(((w[i]-1.)**2. * (1.+10.*(math.sin(math.pi*w[i]+1.)**2.)) for i in range(len(w)))) + \
        (w[-1]-1.)**2. * (1.+(math.sin(2.*math.pi*w[-1]))**2.),
illumination_levy = algorithms.partial(_illuminate_artificial_landscape, func=levy) # type: ignore
class LevyBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-10., 10.)
        super().__init__(fn=algorithms.partial(illumination_levy, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")


def perm0db(ind, beta=10.):
    """Perm function 0, D, BETA defined as:
    $$ f(x) = \sum_{i=1}^n (\sum_{j=1}^n (j+\beta) (x_j^i - \frac{1}{j^i}) )^2$$
    with a search domain of $-n < x_i < n, 1 \leq i \leq n$.
    The global minimum is at $f(x_1, ..., x_n) = f(1, 2, ..., n) = 0.
    """
    return sum(( \
        ( sum(( (j+1.+beta) * ((ind[j])**float(i+1) - 1./(float(j+1)**float(i+1))) for j in range(len(ind)) )) )**2. \
        for i in range(len(ind)) )),
illumination_perm0db = algorithms.partial(_illuminate_artificial_landscape, func=perm0db) # type: ignore
class Perm0dbBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (float(-nb_features), float(nb_features))
        super().__init__(fn=algorithms.partial(illumination_perm0db, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")


def permdb(ind, beta=0.5):
    """Perm function D, BETA defined as:
    $$ f(x) = \sum_{i=1}^d (\sum_{j=1}^n (j^i+\beta) ((\frac{x_j}{j})^i - 1) )^2$$
    with a search domain of $-n < x_i < n, 1 \leq i \leq n$.
    The global minimum is at $f(x_1, ..., x_n) = f(1, 1/2, ..., 1/n) = 0.
    """
    return sum(( \
        ( sum(( ((j+1.)**(i+1.)+beta) * ((ind[j]/(j+1.))**(i+1.) - 1.) for j in range(len(ind)) )) )**2. \
        for i in range(len(ind)) )),
illumination_permdb = algorithms.partial(_illuminate_artificial_landscape, func=permdb) # type: ignore
class PermdbBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (float(-nb_features), float(nb_features))
        super().__init__(fn=algorithms.partial(illumination_permdb, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")



def trid(ind):
    """Trid function defined as:
    $$ f(x) = \sum_{i=1}^n (x_i - 1)^2 - \sum_{i=2}^n x_i x_{i-1}$$
    with a search domain of $-n^2 < x_i < n^2, 1 \leq i \leq n$.
    The global minimum is $f(x^{*}) = -n(n+4)(n-1)/6$ at $x_i = i(n+1-i)$ for all $i=1,2,...,n$.
    """
    return sum(( (x-1)**2. for x in ind)) - sum(( ind[i] * ind[i-1] for i in range(1, len(ind)) )),
illumination_trid = algorithms.partial(_illuminate_artificial_landscape, func=trid) # type: ignore
class TridBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-float(nb_features)**2., float(nb_features)**2.)
        super().__init__(fn=algorithms.partial(illumination_trid, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")


def zakharov(ind):
    """Zakharov function defined as:
    $$ f(x) = \sum_{i=1}^n x_i^2 + (\sum_{i=1}^n 0.5 i x_i)^2 + (\sum_{i=1}^n 0.5 i x_i)^4$$
    with a search domain of $-5 < x_i < 10, 1 \leq i \leq n$.
    The global minimum is at $f(x_1, ..., x_n) = f(0, ..., 0) = 0.
    """
    return  sum((x**2. for x in ind)) + \
            sum(( 0.5 * float(i) * ind[i] for i in range(len(ind)) ))**2. + \
            sum(( 0.5 * float(i) * ind[i] for i in range(len(ind)) ))**4.,
illumination_zakharov = algorithms.partial(_illuminate_artificial_landscape, func=zakharov) # type: ignore
class ZakharovBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-5., 10.)
        super().__init__(fn=algorithms.partial(illumination_zakharov, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")


def dixon_price(ind):
    """Dixon Price function defined as:
    $$ f(x) = (x_0 - 1)^2 + \sum_{i=2}^n i (2 x_i^2 - x_{i-1})^2$$
    with a search domain of $-10 < x_i < 10, 1 \leq i \leq n$.
    The global minimum is $f(x^{*}) = 0$ at $x_i = 2^{-(2^i-2)/(2^i)}$ for all $i=1,2,...,n$.
    """
    return  (ind[0] - 1.)**2. + \
            sum(( float(i) * (2.*(ind[i])**2. - ind[i-1])**2. for i in range(len(ind)) )),
illumination_dixon_price = algorithms.partial(_illuminate_artificial_landscape, func=dixon_price) # type: ignore
class DixonPriceBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-10., 10.)
        super().__init__(fn=algorithms.partial(illumination_dixon_price, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")


def powell(ind):
    """Powell function defined as:
    $$ f(x) = \sum_{i=1}^{n/4} ( (x_{4i-3}+10x_{4i-2})^2. + 5(x_{4i-1}-x_{4i})^2 + (x_{4i-2}-2x_{4i-1})^4 + 10(x_{4i-3}-x{4i})^4 )$$
    with a search domain of $-4 < x_i < 5, 1 \leq i \leq n$.
    The global minimum is at $f(x_1, ..., x_n) = f(0, ..., 0) = 0.
    """
    return  sum(( (ind[4*i-3] + 10.*ind[4*i-2])**2. for i in range(len(ind)//4) )) + \
            sum(( 5.* ((ind[4*i-1] - ind[4*i])**2.) for i in range(len(ind)//4) )) + \
            sum(( (ind[4*i-2] - 2.*ind[4*i-1])**4. for i in range(len(ind)//4) )) + \
            sum(( 10.* ((ind[4*i-3] - ind[4*i])**4.) for i in range(len(ind)//4) )),
illumination_powell = algorithms.partial(_illuminate_artificial_landscape, func=powell) # type: ignore
class PowellBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-4., 5.)
        super().__init__(fn=algorithms.partial(illumination_powell, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")


def michalewicz(ind, m=10.):
    """Michalewicz function defined as:
    $$ f(x) = - \sum_{i=1}^{n} \sin(x_i) \sin( \frac{i x_i^2}{\pi} )^(2*m)$$
    with a search domain of $0 < x_i < \pi, 1 \leq i \leq n$.
    """
    return - sum(( math.sin(ind[i]) * (math.sin(((i+1.)*(ind[i] **2.))/math.pi))**(2.*m) for i in range(len(ind)) )),
illumination_michalewicz = algorithms.partial(_illuminate_artificial_landscape, func=michalewicz) # type: ignore
class MichalewiczBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (0., math.pi)
        super().__init__(fn=algorithms.partial(illumination_michalewicz, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((-math.inf, 0.),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")


def wavy(ind, k=10.):
    """Wavy function defined as:
    $$ f(x) = 1 - \frac{1}{n} \sum_{i=1}^{n} \cos(kx_i)e^{-\frac{x_i^2}{2}}$$
    with a search domain of $-\pi < x_i < \pi, 1 \leq i \leq n$.
    """
    return 1. - sum(( math.cos(k * ind[i]) * math.exp(-(ind[i]*ind[i])/2.) for i in range(len(ind)))) / float(len(ind)),
illumination_wavy = algorithms.partial(_illuminate_artificial_landscape, func=wavy) # type: ignore
class WavyBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-math.pi, math.pi)
        super().__init__(fn=algorithms.partial(illumination_wavy, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., math.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")



def trigonometric02(ind, k=10.):
    """Trigonometric02 function defined as:
    $$ f(x) = 1 - \frac{1}{n} \sum_{i=1}^{n} \cos(kx_i)e^{-\frac{x_i^2}{2}}$$
    with a search domain of $-\pi < x_i < \pi, 1 \leq i \leq n$.
    """
    return 1. + sum((8.*math.sin(7. * (ind[i] - 0.9)**2.)**2. + 6.*math.sin(14.*(ind[i]-0.9)**2.)**2. + (ind[i] - 0.9)**2. for i in range(len(ind)))),
illumination_trigonometric02 = algorithms.partial(_illuminate_artificial_landscape, func=trigonometric02) # type: ignore
class Trigonometric02Benchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-500., 500.)
        super().__init__(fn=algorithms.partial(illumination_trigonometric02, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((1., np.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")



def qing(ind, k=10.):
    """Qing function defined as:
    $$ f(x) = \sum_{i=1}^{n} (x_i^2 - i)^2$$
    with a search domain of $-500 < x_i < 500, 1 \leq i \leq n$.
    """
    return sum(( (ind[i]**2. - i)**2. for i in range(len(ind)))),
illumination_qing = algorithms.partial(_illuminate_artificial_landscape, func=qing) # type: ignore
class QingBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-500., 500.)
        super().__init__(fn=algorithms.partial(illumination_qing, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., np.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")


def small_qing(ind, k=10.):
    """Qing function defined as:
    $$ f(x) = \sum_{i=1}^{n} (x_i^2 - i)^2$$
    with a search domain of $-4 < x_i < 4, 1 \leq i \leq n$.
    """
    return sum(( (ind[i]**2. - i)**2. for i in range(len(ind)))),
illumination_small_qing = algorithms.partial(_illuminate_artificial_landscape, func=small_qing) # type: ignore
class SmallQingBenchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-4., 4.)
        super().__init__(fn=algorithms.partial(illumination_small_qing, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((0., np.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")




def deb01(ind):
    """Deb01 function defined as:
    $$ f(x) = - \frac{1}{n} \sum_{i=1}^{n} \sin(5 \pi x_i)^6 $$
    with a search domain of $-1 < x_i < 1, 1 \leq i \leq n$.
    """
    return -sum(( math.sin(5. * math.pi * ind[i])**6. for i in range(len(ind)))) / float(len(ind)),
illumination_deb01 = algorithms.partial(_illuminate_artificial_landscape, func=deb01) # type: ignore
class Deb01Benchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-1., 1.)
        super().__init__(fn=algorithms.partial(illumination_deb01, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((-1., 1.),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")



def shubert04(ind):
    """Shubert04 function defined as:
    $$ f(x) = \sum_{i=1}^n \sum_{j=1}^5 j cos((j+1)x_i) + j $$
    with a search domain of $-10 < x_i < 10, 1 \leq i \leq n$.
    """
    return sum(( sum((-j * math.cos((j+1) * ind[i] + j) for j in range(1, 6))) for i in range(len(ind)))),
illumination_shubert04 = algorithms.partial(_illuminate_artificial_landscape, func=shubert04) # type: ignore
class Shubert04Benchmark(Benchmark):
    def __init__(self, nb_features: int = 2):
        self.nb_features = nb_features
        ind_domain = (-10., 10.)
        super().__init__(fn=algorithms.partial(illumination_shubert04, nb_features=nb_features),
                ind_domain = ind_domain,
                fitness_domain = ((-29.016015, np.inf),), features_domain = (ind_domain,) * nb_features,
                default_task = "minimisation")





# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
