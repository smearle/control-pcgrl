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

"""Some base classes, stubs and types."""
#from __future__ import annotations

#__all__ = ["jit"]


########### IMPORTS ########### {{{1
from typing import Optional, Tuple, List, Iterable, Iterator, Any, TypeVar, Generic, Union, Sequence, MutableSet, MutableSequence, Type, Callable, Generator, MutableMapping, overload
from typing_extensions import runtime, Protocol
from operator import mul, truediv
import math
import sys
import importlib
import pickle
import textwrap
import copy
from inspect import signature
from functools import partial
import warnings
import numpy as np

import os
import psutil
import signal

from qdpy.utils import *


########### INTERFACES AND STUBS ########### {{{1
T = TypeVar("T")
DomainLike = Tuple[float, float]
ShapeLike = Tuple[int, ...]


########### CONCURRENCY ########### {{{1

@runtime
class FutureLike(Protocol):
    def cancelled(self) -> bool: ...
    def done(self) -> bool: ...
    def result(self) -> Any: ...
    def exception(self) -> Any: ...
    def add_done_callback(self, fn: Callable, **kwargs: Any) -> None: ...
    def as_completed(self, fs: Sequence) -> Any: ...

@runtime
class ExecutorLike(Protocol):
    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> FutureLike: ...
    def map(self, fn: Callable, *iterables: Any) -> Iterable: ...
    def shutdown(self, wait: bool = True) -> None: ...


class SequentialFuture(FutureLike):
    """Future object returned by `SequentialExecutor` methods."""
    _result: Any
    def __init__(self, result: Any) -> None:
        self._result = result
    def cancelled(self) -> bool:
        return False
    def done(self) -> bool:
        return True
    def result(self) -> Any:
        return self._result
    def exception(self) -> Any:
        return None
    def add_done_callback(self, fn: Callable, **kwargs: Any) -> None:
        # Call directly the callback
        fn()

class SequentialExecutor(ExecutorLike):
    """Executor that runs sequentially (no parallelism)."""
    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> FutureLike:
        return SequentialFuture(fn(*args, **kwargs))
    def map(self, fn: Callable, *iterables: Any) -> Iterable:
        # From cpython code 'concurrent/futures/_base.py'
        fs = [self.submit(fn, *args) for args in zip(*iterables)]
        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def result_iterator():
            try:
                # reverse to keep finishing order
                fs.reverse()
                while fs:
                    # Careful not to keep a reference to the popped future
                    yield fs.pop().result()
            finally:
                for future in fs:
                    future.cancel()
        return result_iterator()
    def shutdown(self, wait: bool = True) -> None:
        pass


try:
    import scoop.futures
    class ScoopExecutor(ExecutorLike):
        """Executor that encapsulate scoop concurrency functions. Need the scoop package to be installed and importable."""
        def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> FutureLike:
            assert len(kwargs) == 0, f"`ScoopExecutor.submit` does not handle **kwargs."
            return scoop.futures.submit(fn, *args)
        def map(self, fn: Callable, *iterables: Any) -> Iterable:
            return scoop.futures.map(fn, *iterables)
        def shutdown(self, wait: bool = True) -> None:
            scoop.futures.shutdown(wait)
    _scoop_as_completed: Optional[Callable] = scoop.futures.as_completed
    _scoop_future_class: Optional[Any] = scoop.futures.Future

except ImportError:
    class ScoopExecutor(ExecutorLike): # type: ignore
        """Executor that encapsulate scoop concurrency functions. Need the scoop package to be installed and importable."""
        def __init__(self) -> None:
            raise NotImplementedError("`ScoopExecutor` needs the scoop package to be installed and importable.")
        def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> FutureLike:
            raise NotImplementedError("`ScoopExecutor` needs the scoop package to be installed and importable.")
        def map(self, fn: Callable, *iterables: Any) -> Iterable:
            raise NotImplementedError("`ScoopExecutor` needs the scoop package to be installed and importable.")
        def shutdown(self, wait: bool = True) -> None:
            raise NotImplementedError("`ScoopExecutor` needs the scoop package to be installed and importable.")
    _scoop_as_completed: Optional[Callable] = None # type: ignore
    _scoop_future_class: Optional[Any] = None # type: ignore



try:
    import ray

    class RayFuture(FutureLike):
        """Future object returned by `RayExecutor` methods."""
        oid: Any
        def __init__(self, oid: Any) -> None:
            self.oid = oid
        def cancelled(self) -> bool:
            return False # TODO
        def done(self) -> bool:
            return len(ray.wait([self.oid], 1)[0]) == 1
        def result(self) -> Any:
            return ray.get(self.oid)
        def exception(self) -> Any:
            return None # TODO
        def add_done_callback(self, fn: Callable, **kwargs: Any) -> None:
            # TODO
            # Call directly the callback
            fn()

    @ray.remote
    def _ray_deploy_func(func, kwargs, *args):
        return func(*args, **kwargs)

    @ray.remote
    class RayDeployment(object):
        def __init__(self, i):
            pass # TODO
        def ray_deploy_func(self, func, kwargs, *args):
            return func(*args, **kwargs)

    class RayExecutor(ExecutorLike): # type: ignore
        """Executor that encapsulate ray concurrency functions. Need the ray package to be installed and importable."""
        def __init__(self) -> None:
            num_cpus = psutil.cpu_count(logical=True)
            ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
            #self.nb_actors = num_cpus
            #self.actors = [RayDeployment.remote(i) for i in range(self.nb_actors)]
            #self.current_actor_idx = 0
        def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> FutureLike:
            return RayFuture(_ray_deploy_func.remote(fn, kwargs, *args)) # TODO
            #future = RayFuture(self.actors[self.current_actor_idx].ray_deploy_func.remote(fn, kwargs, *args))
            #self.current_actor_idx = (self.current_actor_idx + 1) % self.nb_actors
            #return future
        def map(self, fn: Callable, *iterables: Any) -> Iterable:
            # From cpython code 'concurrent/futures/_base.py'
            fs = [self.submit(fn, *args) for args in zip(*iterables)]
            # Yield must be hidden in closure so that the futures are submitted
            # before the first iterator value is required.
            def result_iterator():
                try:
                    # reverse to keep finishing order
                    fs.reverse()
                    while fs:
                        # Careful not to keep a reference to the popped future
                        yield fs.pop().result()
                finally:
                    for future in fs:
                        future.cancel()
            return result_iterator()
        def shutdown(self, wait: bool = True) -> None:
            ray.shutdown()

    def _ray_as_completed_fn(fs, timeout=None):
        oid_lst = [x.oid for x in fs]
        oid_dict = {x.oid: x for x in fs}
        for _ in range(len(fs)):
            ready_lst, remaining_lst = ray.wait(oid_lst, 1, timeout=timeout)
            yield oid_dict[ready_lst[0]]
    _ray_as_completed: Optional[Callable] = _ray_as_completed_fn
    _ray_future_class: Optional[Any] = RayFuture


except ImportError:
    class RayExecutor(ExecutorLike): # type: ignore
        """Executor that encapsulate ray concurrency functions. Need the ray package to be installed and importable."""
        def __init__(self) -> None:
            raise NotImplementedError("`RayExecutor` needs the ray package to be installed and importable.")
        def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> FutureLike:
            raise NotImplementedError("`RayExecutor` needs the ray package to be installed and importable.")
        def map(self, fn: Callable, *iterables: Any) -> Iterable:
            raise NotImplementedError("`RayExecutor` needs the ray package to be installed and importable.")
        def shutdown(self, wait: bool = True) -> None:
            raise NotImplementedError("`RayExecutor` needs the ray package to be installed and importable.")
    _ray_as_completed: Optional[Callable] = None # type: ignore
    _ray_future_class: Optional[Any] = None # type: ignore



def generic_as_completed(fs: Sequence[FutureLike], timeout: Optional[int]=None) -> FutureLike:
    """Generic `as_completed` function, that can work with any kind of FutureLike object, including
    Futures from `concurrent.futures` and `scoop.futures`.
    Assume that all items from `fs` are instance of the same `FutureLike` class."""
    if len(fs) == 0:
        raise ValueError(f"`fs` must not be empty.")
    elif len(fs) == 1:
        _ = fs[0].result()
        return fs[0]
    # Verify if the futures are instances of SequentialFuture (assume all items have the same class)
    if isinstance(fs[0], SequentialFuture):
        return fs[0]
    if _scoop_future_class is not None and isinstance(fs[0], _scoop_future_class) and _scoop_as_completed is not None:
        return next(_scoop_as_completed(fs, timeout))
    if _ray_future_class is not None and isinstance(fs[0], _ray_future_class) and _ray_as_completed is not None:
        return next(_ray_as_completed(fs, timeout))
    # Get the class and module of the first item (assume all items have the same class)
    futures_class = fs[0].__class__
    futures_module = fs[0].__module__
    # Verify if a `as_completed` function is already available
    m = importlib.import_module(futures_module)
    if hasattr(m, "as_completed"):
        return next(m.as_completed(fs, timeout)) # type: ignore
    else:
        raise ValueError(f"Futures from `fs` are of unknown type, with unknown `as_completed` corresponding function.")




# Hack to terminate child process/threads gracefully using Ctrl-C
# From https://stackoverflow.com/questions/32160054/keyboard-interrupts-with-pythons-multiprocessing-pool-and-map-function/45259908
_parent_id = os.getpid()
def _worker_init():
#    signal.signal(signal.SIGINT, signal.SIG_IGN)
    def sig_int(signal_num, frame):
        #print('signal: %s' % signal_num)
        parent = psutil.Process(_parent_id)
        for child in parent.children():
            if child.pid != os.getpid():
                print("killing child: %s" % child.pid)
                child.kill()
        #print("killing parent: %s" % _parent_id)
        parent.kill()
        #print("suicide: %s" % os.getpid())
        psutil.Process(os.getpid()).kill()
    signal.signal(signal.SIGINT, sig_int)


# TODO Support for Dask
class ParallelismManager(object):
    """Simplify parallelism handling by generating on-the-fly executor and toolboxes to conform to a
    specified kind of parallelism. It can be used with qdpy QDAlgorithm classes, or with DEAP toolboxes.
    It currently handle three kind of parallelism: Sequential (no parallelism), Python PEP3148 concurrent.futures,
    and SCOOP (https://github.com/soravux/scoop)."""

    executor: ExecutorLike
    toolbox: Optional[Any]
    orig_toolbox: Optional[Any]
    max_workers: Optional[int]
    _parallelism_type: str = ""
    _parallelism_type_list: Sequence[str] = ['none', 'sequential', 'multiprocessing', 'concurrent', 'multithreading', 'scoop', 'ray']
    force_worker_suicide: bool

    def __init__(self, parallelism_type: str = "multiprocessing", max_workers: Optional[int] = None,
            toolbox: Optional[Any] = None, force_worker_suicide: bool = True) -> None:
        self.parallelism_type = parallelism_type
        self.max_workers = max_workers
        self.orig_toolbox = toolbox
        self.toolbox = copy.deepcopy(toolbox) if toolbox is not None else None
        self.force_worker_suicide = force_worker_suicide

    @property
    def parallelism_type(self) -> str:
        return self._parallelism_type

    @parallelism_type.setter
    def parallelism_type(self, val: str) -> None:
        if len(self._parallelism_type) > 0:
            raise NotImplementedError("`parallelism_type` cannot be set after initialisation.")
        lval = val.lower()
        if not lval in self._parallelism_type_list:
            raise ValueError(f"`parallelism_type` can only be from [{','.join(self._parallelism_type_list)}].")
        self._parallelism_type = lval

    def __enter__(self) -> Any:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self) -> Any:
        """Open and prepare a new executor according to `self.parallelism_type`."""
        if self.force_worker_suicide:
            init_fn: Optional[Callable] = _worker_init
        else:
            init_fn = None
        if self.parallelism_type == "none" or self.parallelism_type == "sequential":
            self.executor = SequentialExecutor()
        elif self.parallelism_type == "multiprocessing" or self.parallelism_type == "concurrent":
            import concurrent.futures
            if sys.version_info >= (3, 7):
                import multiprocessing
                ctx = multiprocessing.get_context("forkserver")
                self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx, initializer=init_fn) # type: ignore
            else:
                self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) # type: ignore
        elif self.parallelism_type == "multithreading":
            import concurrent.futures
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) # type: ignore
        elif self.parallelism_type == "scoop":
            import scoop
            self.executor = ScoopExecutor()
        elif self.parallelism_type == "ray":
            import ray
            self.executor = RayExecutor()
        else:
            raise ValueError("Unknown parallelism_type: '%s'" % self.parallelism_type)
        if self.toolbox is not None:
            self.toolbox.register("map", self.executor.map)
        return self

    def close(self) -> None:
        """Close the current executor."""
        self.executor.shutdown()



########### BASE CLASSES ########### {{{1
class Summarisable(object):
    """Describes a class that can be summarised by using the `self.summary` method.
    The summarised information is provided by the `self.__get_summary_state__` method. """


    def __get_summary_state__(self) -> Mapping[str, Any]:
        """Return a dictionary containing the relevant entries to build a summary of the class.
        By default, it includes all public attributes of the class. Must be overridden by subclasses."""
        # Find public attributes
        entries = {}
        for k, v in inspect.getmembers(self):
            if not k.startswith('_') and not inspect.ismethod(v):
                try:
                    entries[k] = v
                except Exception:
                    pass
        return entries


    def summary(self, max_depth: Optional[int] = None, max_entry_length: Optional[int] = 250) -> str:
        """Return a summary description of the class.
        The summarised information is provided by the `self.__get_summary_state__` method.

        Parameters
        ----------
        :param max_depth: Optional[int]
            The maximal recursion depth allowed. Used to summarise attributes of `self` that are also Summarisable.
            If the maximal recursion depth is reached, the attribute is only described with a reduced representation (`repr(attribute)`).
            If `max_depth` is set to None, there are no recursion limit.
        :param max_entry_length: Optional[int]
            If `max_entry_length` is not None, the description of a non-Summarisable entry exceeding `max_entry_length`
            is cropped to this limit.
        """
        res: str = f"Summary {self.__class__.__name__}:\n"
        subs_max_depth = max_depth - 1 if max_depth is not None else None
        summary_state = self.__get_summary_state__()
        for i, k in enumerate(summary_state.keys()):
            v = summary_state[k]
            res += f"  {k}:"
            if isinstance(v, Summarisable):
                if max_depth is None or max_depth > 0:
                    res += textwrap.indent(v.summary(subs_max_depth), '  ')
                else:
                    res += f" {repr(v)}"
            else:
                str_v = f" {v}"
                str_v = str_v.replace("\n", " ")
                if max_entry_length is not None and len(str_v) > max_entry_length:
                    str_v = str_v[:max_entry_length - 4] + " ..."
                res += str_v
            if i != len(summary_state) - 1:
                res += "\n"
        return res

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"



class Saveable(object):
    """Describes a class with internal information that can be saved into an output file.
    The list of attributes from `self` that are saved in the output file are provided by the method `self.__get_saved_state__`.
    """

    def __get_saved_state__(self) -> Mapping[str, Any]:
        """Return a dictionary containing the relevant information to save.
        By default, it includes all public attributes of the class. Must be overridden by subclasses."""
        # Find public attributes
        entries = {k:v for k,v in inspect.getmembers(self) if not k.startswith('_') and not inspect.ismethod(v)}
        return entries

    def save(self, output_path: str, output_type: str = "pickle") -> None:
        """Save the into an output file.

        Parameters
        ----------
        :param output_path: str
            Path of the output file.
        :param output_type: str
            Type of the output file. Currently, only supports "pickle".
        """
        if output_type != "pickle":
            raise ValueError(f"Invalid `output_type` value. Currently, only supports 'pickle'.")
        saved_state = self.__get_saved_state__()
        with open(output_path, "wb") as f:
            pickle.dump(saved_state, f)


class Copyable(object):
    """Describes a class capable to be copied and deepcopied."""

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result



class CreatableFromConfig(object):
    """Describe a class capable to be created from a configuration dictionary."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> Any:
        """Create a class using the information from `config` to call the class constructor.

        Parameters
        ----------
        :param config: Mapping[str, Any]
            The configuration mapping used to create the class. For each entry, the key corresponds
            to the name of a parameter of the constructor of the class.
        :param kwargs: Any
            Additional information used to create the class. The configuration entries from kwargs
            take precedence over the entry in `config`.
        """
        final_kwargs = {**config, **kwargs}
        return cls(**final_kwargs) # type: ignore


# Inspired from Nevergrad (MIT License) Registry class (https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/common/decorators.py)
class Registry(dict):
    """Registers function or classes as a dict."""

    def __init__(self) -> None:
        super().__init__()
        self._information: MutableMapping[str, Mapping] = {}

    def register(self, obj: Any, info: Optional[Mapping[Any, Any]] = None) -> Any:
        """Decorator method for registering functions/classes
        The `info` variable can be filled up using the register_with_info
        decorator instead of this one.
        """
        name = obj.__name__
        if name in self:
            #raise RuntimeError(f'Encountered a name collision "{name}".')
            warnings.warn(f'Encountered a name collision "{name}".')
            return self[name]
        self[name] = obj
        if info is not None:
            self._information[name] = info
        return obj

    def get_info(self, name: str) -> Mapping[str, Any]:
        if name not in self:
            raise ValueError(f'`{name}` is not registered.')
        return self._information.setdefault(name, {})

registry = Registry()


class Factory(dict):
    """Build objects from configuration."""

    def __init__(self, registry: Registry=registry) -> None:
        super().__init__()
        self.registry = registry


    def _get_name(self, obj: Any, key: Optional[str], config: Mapping[str, Any]) -> Optional[str]:
        """Check if `obj` possess a name, and return it.""" 
        name: str = ""
        if hasattr(obj, "name"):
            return obj.name
        elif key is not None:
            return key
        elif "name" in config:
            return config["name"]
        else:
            return None


    def _build_internal(self, config: MutableMapping[str, Any], default_params: Mapping[str, Any] = {}, **kwargs: Any) -> Any:
        # Recursively explore config and create objects for each entry possessing a "type" configuration entry
        built_objs = []
        default_config = {**default_params} 
        for k, v in config.items():
            if is_iterable(v) and "type" in v:
                sub_obj = self._build_internal(v, {**default_config, "name": k})
                sub_name = self._get_name(sub_obj, k, v)
                if sub_name is not None and len(sub_name) > 0 and not sub_name in self:
                    self[sub_name] = sub_obj
                    built_objs.append(sub_obj)
                config[k] = sub_obj
            elif isinstance(v, str) and v in self:
                config[k] = self[v]
            elif isinstance(v, Mapping):
                v_ = copy.copy(v)
                config[k] = []
                for k2, v2 in v_.items():
                    if isinstance(k2, str) and isinstance(v2, int) and k2 in self:
                        for _ in range(v2):
                            new_k2 = copy.deepcopy(self[k2])
                            new_k2.reinit()
                            config[k].append(new_k2)
            elif is_iterable(v):
                for i, val in enumerate(v):
                    if isinstance(val, str) and val in self:
                        new_val = copy.deepcopy(self[val])
                        new_val.reinit()
                        config[k][i] = new_val
            if not is_iterable(v) or (is_iterable(v) and "type" not in v):
                default_config[k] = config[k]

        # If the configuration describes an object, build it
        if "type" in config:
            # Retrieve the class from registry
            #assert "type" in config, f"The configuration must include an entry 'type' containing the type of the object to be built."
            type_name: str = config["type"]
            assert type_name in self.registry, f"The class `{type_name}` is not declared in the registry. To get the list of registered classes, use: 'print(registry.keys())'."
            cls: Any = self.registry[type_name]
            assert issubclass(cls, CreatableFromConfig), f"The class `{cls}` must inherit from `CreatableFromConfig` to be built."
            # Build object
            obj: Any = cls.from_config(config, **default_params, **kwargs)
            # If 'obj' possess a name, add it to the factory
            name: Optional[str] = self._get_name(obj, None, config)
            if name is not None and len(name) > 0 and not name in self:
                self[name] = obj
            return obj

        else: # Return all built objects
            return built_objs



    def build(self, config: Mapping[str, Any], default_params: Mapping[str, Any] = {}, **kwargs: Any) -> Any:
        """Create and return an object from `self.registry` based on
        configuration entries from `self` and `config`. The class of the created
        object must inherit from the Class `CreatableFromConfig`.
        The object is created by iteratively executing the `from_config` methods.
        The class of the object must be specified in configuration entry 'type'.

        If a configuration entry contains a sub-entry 'type', it also created by the factory.
        If it also contains a sub-entry 'name', it is added to `self` with key 'name', and accessible
        through `self[name]`.
        
        Parameters
        ----------
        :param config: Mapping[str, Any]
            The mapping containing configuration entries of the built object.
        :param default_params:
            Additional configuration entries to send to the `from_config` class method when creating the main object and all sub-objects.
        :param kwargs: Any
            Additional configuration entries to send to the `from_config` class method when creating the main object.
        """
        if "name" in config and config["name"] in self:
            return self[config["name"]]
        final_config: MutableMapping[str, Any] = dict(copy.deepcopy(config))
        return self._build_internal(final_config, default_params, **kwargs)



from qdpy.phenotype import *

# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
