# STM Simulation Script

A Python tool for generating and visualising synthetic scanning‑tunnelling‑microscope (STM) images of crystalline surfaces—silicon (Si (111)), graphene, graphite, and gold (Au (111))—in both pristine and defect‑rich forms.

---

## Table of contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Configuration](#configuration)
6. [Extending the script](#extending-the-script)
7. [Notes on physical accuracy](#notes-on-physical-accuracy)
8. [License](#license)

---

## Overview

`STM ver 2.py` is intended for students, researchers, and hobbyists in surface science, nanotechnology, or computational physics who wish to explore how lattice geometry and point defects influence STM contrast.
The code is **object‑oriented and modular**, cleanly separating:

* lattice construction
* defect generation
* tunnelling‑current calculation
* visualisation with `matplotlib`

Although the underlying physics is simplified, the implementation supports rapid, qualitative exploration of many scenarios.

---

## Features

| Category                  | Description                                                                                                                                                                       |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Lattice generation**    | Built‑in support for Si (111), graphene, graphite, and Au (111) surfaces. Custom lattice constants, basis atoms, unit‑cell counts, and layer numbers (graphite) are configurable. |
| **Defect modelling**      | Random vacancies and substitutional/positional impurities with independent probabilities and displacement amplitudes.                                                             |
| **STM image synthesis**   | Exponential tunnelling‑current model with adjustable tip height, decay constant, scan size, pixel resolution, and Gaussian noise.                                                 |
| **Visualisation**         | Side‑by‑side plots: atomic lattice (scatter) and STM map (heatmap).                                                                                                               |
| **Interactive interface** | `main_menu()` prompts for all parameters—no code editing required.                                                                                                                |

---

## Installation

```bash
git clone https://github.com/your‑username/your‑repo.git
cd your‑repo
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install numpy matplotlib
```

---

## Quick start

```bash
python "STM ver 2.py"
```

1. Select a lattice type from the menu.
2. Accept defaults or enter new values.
3. View the generated lattice and its STM image.

---

## Configuration

All run‑time options are stored in the top‑level `CONFIG` dictionary.

| Group       | Key                  | Default  | Meaning                                |
| ----------- | -------------------- | -------- | -------------------------------------- |
| **Lattice** | `a`                  | *varies* | Lattice spacing (nm).                  |
|             | `c`                  | *varies* | Inter‑layer spacing for graphite (nm). |
|             | `nx`, `ny`           | `10`     | Unit cells along x and y.              |
|             | `nlayers`            | `3`      | Graphite only.                         |
| **Defects** | `vacancy_prob`       | `0.03`   | Fraction of missing atoms.             |
|             | `impurity_prob`      | `0.05`   | Fraction of displaced atoms.           |
|             | `displacement_scale` | `0.05`   | Maximum impurity displacement (nm).    |
| **STM**     | `resolution`         | `0.02`   | Pixel size (nm ∕ px).                  |
|             | `tip_height`         | `0.35`   | Tip–sample distance (nm).              |
|             | `decay_const`        | `10.0`   | Tunnelling decay constant (nm⁻¹).      |
|             | `noise_level`        | `0.05`   | Added Gaussian noise (relative).       |

### Example: denser defects and finer imaging

```python
CONFIG.update({
    "vacancy_prob":       0.10,
    "impurity_prob":      0.12,
    "resolution":         0.01,
    "tip_height":         0.30,
    "decay_const":        11.0,
})
```

---

## Extending the script

* **New materials** → subclass `Lattice` with custom lattice vectors and basis.
* **Additional defect types** → add methods to `introduce_vacancies` / `introduce_impurities`.
* **Alternative visualisations** → modify `plot_lattice` or `plot_stm_comparison`.

---

## Notes on physical accuracy

The model focuses on qualitative trends. It omits detailed electronic structure, tip states, and many‑body effects; consequently, absolute currents, corrugation amplitudes, and bias‑dependent contrast are **not** quantitatively reliable. Treat output as an educational guide rather than a predictive calculation.

---

## License

Distributed under the MIT License. See `LICENSE` for full text.

