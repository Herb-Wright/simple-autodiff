
# Simple Autodiff

## Overview

This is a simple package for automatic differentiation. It is not meant to be performant and is instead meant to illustrate the simplicity and power of auto diff. This package is built on top of numpy and is inspired by the autograd engine in pytorch. I decided to code this because in my ML class, we are not supposed to use pytorch, so I coded this myself so I can use autodiff without getting into trouble. 

The `./autodiff` folder contains the main code for the package. The `./examples` folder contains examples of how you could use the package. The `./tests` folder contains tests to verify that the autodiff calculations are correct.

## Getting Started

1. install numpy in your python environment
2. clone the repo
    ```sh
    git clone https://github.com/Herb-Wright/simple-autodiff.git
    cd simple-autodiff
    ```
3. install the project:
    ```sh
    pip install -e .
    ```
4. use the project
    - in your own code: use `import simple_autodiff as ad`
    - in the examples folder (have to install `tqdm`) 


## Dependencies

For the package:

- `python3`
- `numpy`

Dependencies for the examples:

- `tqdm`

Dependencies specific to tests:

- `pytorch`


## Cooler Packages than This

- [pytorch](https://pytorch.org/)
- [numpy](https://numpy.org/)
