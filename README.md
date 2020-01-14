# LP-SparseMAP
Differentiable sparse structured prediction in coarse factor graphs


This repo contains:

  - `ad3qp`: an updated fork of [`ad3`](https://github.com/andre-martins/ad3),
  supporting the solving of SparseMAP QPs in arbitrary factor graphs. (C++, LGPL
  license.)

  - `dysparsemap`: a library that provides a dynet function using `ad3qp` for
  forward and backward pass computation for structured hidden layers. (C++, MIT
  license.) 

  - `lpsmap`: a python wrapper for `ad3qp` and some example usage scripts.
  (cython and python, MIT license.)  This repository is a work-in-progress,
  with the end-goal to drastically simplify the AD3 API.
  

## Reference

Vlad Niculae and Andre F. T. Martins.
LP-SparseMAP: Differentiable Relaxed Optimization for Sparse Structured
Prediction. https://arxiv.org/abs/2001.04437


## lpsmap

*Requirements:*
 - Cython
 - [Eigen](https://gitlab.com/libeigen/eigen)

For examples and tests: numpy, pytest.

*Installation:*

```
export EIGEN_DIR=/path/to/eigen
pip install .              # builds ad3, lpsmap and installs
```

*In-place installation:*

```
export EIGEN_DIR=/path/to/eigen
python setup.py build_clib  # builds ad3 in-place
pip install -e .            # builds lpsmap and creates a link
```


## dysparsemap

Installation uses cmake.  Examples to come soon.
