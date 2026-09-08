# Multidirectional Height-Field Topology Optimisation

This repository contains the Python implementation developed for an MSc research project on subtractive-manufacturing-aware topology optimisation.

## Overview

The project investigates how machining direction selection affects topology optimisation results using multidirectional height-field parameterisation. The framework is implemented with JAX and JAX-FEM and uses gradient-based optimisation with the Method of Moving Asymptotes (MMA).

The repository includes:

* `TrueDimension_beam_simp.py` — conventional density-based SIMP topology optimisation baseline
* `Cartesian.py` — topology optimisation with selectable Cartesian cutting directions
* `arbitrary_field.py` — topology optimisation with arbitrary prescribed cutting directions

## Installation

Create the Conda environment using:

```bash
conda env create -f environment.yml
conda activate research
```

## Usage

Run a case from the repository root, for example:

```bash
python src/TrueDimension_beam_simp.py
```

or

```bash
python src/Cartesian.py
```

Generated optimisation results are stored locally in the `outputs/` directory.


## Author

Yu He
MSc research project, Imperial College London
