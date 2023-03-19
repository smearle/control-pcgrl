from setuptools import find_namespace_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='control_pcgrl',
      version='0.5.0',
      author="NYU Game Innovation Lab",
      author_email="sam.earle@nyu.edu",
      description="A package for controllable \"Procedural Content Generation via Reinforcement Learning\" OpenAI Gym interface.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/smearle/gym-pcgrl",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      # packages=['control_pcgrl'],
      packages=find_namespace_packages(include=["hydra_plugins.*", "control_pcgrl"]),
      scripts=[
        'bin/train_pcgrl', 
        'bin/enjoy_pcgrl',
        'bin/eval_pcgrl',
        'bin/cross_eval_pcgrl', 
      ],
)
