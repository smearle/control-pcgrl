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

"""TODO""" # TODO

__all__ = ["SurrogateAssistedAlg", "SAIL"]


########### IMPORTS ########### {{{1

import abc
from timeit import default_timer as timer
from typing import Optional, Tuple, List, Iterable, Iterator, Any, TypeVar, Generic, Union, Sequence, MutableSet, MutableSequence, Type, Callable, Generator, Mapping, MutableMapping, overload
import warnings
import numpy as np
import random
import copy
from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor


from .base import *
from .evolution import *
from qdpy.utils import *
from qdpy.phenotype import *
from qdpy.base import *
from qdpy.containers import *
from qdpy import tools


########### SURROGATE-ASSISTED ALGORITHMS ########### {{{1


class SurrogateAssistedAlg(AlgWithBatchSuggestions):
    """TODO"""

    _illumination_algo: QDAlgorithm
    _acquisition_algo: QDAlgorithm
    _prediction_algo: Optional[QDAlgorithm]
    max_selected_acquisition_suggestions: float # float instead of int to allow `np.inf`
    budget_not_suggested_by_surrogate: float # float instead of int to allow `np.inf`


    def __init__(self,
            illumination_algo: QDAlgorithm,
            acquisition_algo: QDAlgorithm,
            prediction_algo: Optional[QDAlgorithm] = None,
            max_selected_acquisition_suggestions: float = np.inf,
            budget_not_suggested_by_surrogate: float = np.inf,
            **kwargs):
        self._illumination_algo = illumination_algo
        self._acquisition_algo = acquisition_algo
        self._prediction_algo = prediction_algo if prediction_algo else copy.deepcopy(self._acquisition_algo)
        self.max_selected_acquisition_suggestions = max_selected_acquisition_suggestions
        self.budget_not_suggested_by_surrogate = budget_not_suggested_by_surrogate

        kwargs2 = copy.deepcopy(kwargs)
        def del_if_exists(keys: Sequence[str]) -> None:
            for k in keys:
                if k in kwargs2:
                    del kwargs2[k]
        del_if_exists(["container", "budget", "dimension", "batch_size", "nb_objectives", "optimisation_task", "base_ind_gen"])

        super().__init__(illumination_algo.container, illumination_algo.budget,
                dimension=illumination_algo.dimension, batch_size=illumination_algo.batch_size,
                nb_objectives=illumination_algo._nb_objectives, optimisation_task=illumination_algo.optimisation_task,
                base_ind_gen=illumination_algo._base_ind_gen, **kwargs2)


    @property
    def illumination_algo(self) -> QDAlgorithm:
        return self._illumination_algo

    @property
    def acquisition_algo(self) -> QDAlgorithm:
        return self._acquisition_algo

    @property
    def prediction_algo(self) -> Optional[QDAlgorithm]:
        return self._prediction_algo


    @abc.abstractmethod
    def _compute_acquisition(self) -> List[IndividualLike]:
        """Compile and return acquisition suggestions."""
        raise NotImplementedError("`self._compute_acquisition` must be implemented by sub-classes")


    def _select_suggestions_from_acquisition(self, acquisition_suggestions: List[IndividualLike],
            nb_suggestions: int) -> List[IndividualLike]:
        """Select `nb_suggestions` from `acquisition_suggestions` to be used as suggestions for `self._illumination_algo`.
        By default, select randomly (uniformly) `self.max_selected_acquisition_suggestions` individuals among `acquisition_suggestions`.

        Parameters
        ----------
        :param acquisition_suggestions: Sequence[IndividualLike]
            The sequence of proposed suggestions from acquisition.
        :param nb_suggestions: int
            The maximal number of suggestions to include in the resulting sequence of suggestions.
        """
        nb = int(min(nb_suggestions, self.max_selected_acquisition_suggestions))
        if len(acquisition_suggestions) <= nb:
            return acquisition_suggestions
        else:
            random.shuffle(acquisition_suggestions)
            return acquisition_suggestions[:nb]


    def _generate_batch_suggestions(self) -> None:
        """Compute all suggestions for the current batch and store them in `self._batch_inds`."""
        # Compute acquisition if the container is not empty
        acquisition_suggestions: List[IndividualLike] = self._compute_acquisition() if len(self.container) > 0 else []
        # Select suggestions from the acquisition container
        self._batch_inds.extend(self._select_suggestions_from_acquisition(acquisition_suggestions, self.batch_size))
        # If not enough suggestions were obtained from acquisition, ask `self._illumination_algo` for suggestions.
        nb_not_suggested_by_surrogate = int(min(self.budget_not_suggested_by_surrogate, self.batch_size - len(self._batch_inds)))
        for _ in range(nb_not_suggested_by_surrogate):
            self._batch_inds.append(self._illumination_algo.ask())


    def _internal_tell(self, individual: IndividualLike, added_to_container: bool, xattr: Mapping[str, Any] = {}) -> None:
        self._illumination_algo._tell_after_container_update(individual, added_to_container, xattr)



@registry.register
class SAIL(SurrogateAssistedAlg):
    """TODO"""

    ucb_stddev: float
    _gp: Any

    def __init__(self, ucb_stddev: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.ucb_stddev = ucb_stddev
        self._gp = None

    def __getstate__(self):
        odict = super().__getstate__()
        del odict['_gp']
        return odict


    def _acquisition_eval_fn(self, inds: Sequence[IndividualLike]) -> Sequence[Tuple[FitnessValuesLike, FeaturesValuesLike]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_mean, y_std = self._gp.predict(np.array(inds), return_std = True)
        ucb_val = y_mean + self.ucb_stddev * y_std.reshape((-1, 1))
        res = []
        for i in range(len(inds)):
            fitness = []
            for j in range(self._nb_objectives):
                fitness.append(ucb_val[i, j])
            features = []
            for j in range(ucb_val.shape[1] - self._nb_objectives):
                features.append(ucb_val[i, j + self._nb_objectives])
            res.append((fitness, features))
        return res


    def _create_gaussian_regressor(self) -> None:
        # Create dataset
        dataX = []
        dataY = []
        for ind in self.container:
            dataX.append(ind)
            dataY.append(list(ind.fitness.values) + list(ind.features.values))
        # Fit gaussian process on the dataset
        self._gp = GaussianProcessRegressor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(dataX, dataY)


    def _compute_acquisition(self) -> List[IndividualLike]:
        """Compile and return acquisition suggestions."""
        # Reinit `self._acquisition_algo`
        self._acquisition_algo.container.clear()
        # TODO reinit ??

        # Create gaussian regressor
        self._create_gaussian_regressor()
        # Run `self._acquisition_algo` optimisation process
        try:
            self._acquisition_algo.optimise(self._acquisition_eval_fn, executor=None, send_several_suggestions_to_fn=True)
        except RuntimeError:
            pass
        # Retrieve new suggestions
        suggestions = copy.deepcopy(list(set(self._acquisition_algo.container) - set(self.container)))
        for ind in suggestions:
            ind.reset()
            #del ind.fitness.values
            #ind.features = ()
        #print("DEBUG _compute_acquisition", len(suggestions))
        return suggestions


    # TODO sobol
    def _select_suggestions_from_acquisition(self, acquisition_suggestions: List[IndividualLike],
            nb_suggestions: int) -> List[IndividualLike]:
        """Select `nb_suggestions` from `acquisition_suggestions` to be used as suggestions for `self._illumination_algo`.
        By default, select randomly (uniformly) `self.max_selected_acquisition_suggestions` individuals among `acquisition_suggestions`.

        Parameters
        ----------
        :param acquisition_suggestions: Sequence[IndividualLike]
            The sequence of proposed suggestions from acquisition.
        :param nb_suggestions: int
            The maximal number of suggestions to include in the resulting sequence of suggestions.
        """
        nb = int(min(nb_suggestions, self.max_selected_acquisition_suggestions))
        if len(acquisition_suggestions) <= nb:
            return acquisition_suggestions
        else:
            random.shuffle(acquisition_suggestions)
            return acquisition_suggestions[:nb]




# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
