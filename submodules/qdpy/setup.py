#!/usr/bin/env python3


from setuptools import setup, find_packages
import qdpy

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

extras_require = {
    'all': [
        'deap>=1.2.2',
        'tqdm>=4.28.1',
        'colorama>=0.4.1',
        'tables>=3.4.4',
        'cma>=2.6.0',
        'ray>=0.7.4',
    ],
    'deap': [
        'deap>=1.2.2',
    ]
}

packages = find_packages(exclude=['examples'])

setup(name='qdpy',
      long_description=qdpy.__doc__,
      version=qdpy.__version__,
      description='Quality-Diversity algorithms in Python',
      url='https://gitlab.com/leo.cazenille/qdpy',
      author='Leo Cazenille',
      author_email='leo.cazenille@gmail.com',
      license='LGPLv3',
      packages=packages,
      python_requires=">=3.6",
      install_requires=requirements,
      extras_require=extras_require,
      zip_safe=False,
      classifiers = [
            "Intended Audience :: Science/Research",
            "Intended Audience :: Education",
            "Intended Audience :: Other Audience",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Visualization",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Development Status :: 2 - Pre-Alpha",
            "Environment :: Console",
            "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        ],
      keywords=["optimisation", "optimization", "illumination", "Quality-Diversity", "Landscape exploration", "MAP-Elites", "NSLC", "CVT-MAP-Elites", "SAIL", "evolutionary algorithms", "random search"]
      )

# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
