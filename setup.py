from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gym_pcgrl',
      version='0.3.5',
      install_requires=['gym', 'numpy>=1.10', 'pillow',#'baselines',
          'stable_baselines'],
      author="Ahmed Khalifa",
      author_email="ahmed@akhalifa.com",
      description="A package for \"Procedural Content Generation via Reinforcement Learning\" OpenAI Gym interface.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/smearle/gym-pcgrl-sc",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ]
)
