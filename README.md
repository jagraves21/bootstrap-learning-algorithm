# Bootstrap Learning Algorithm (R Implementation)

This repository contains an R reimplementation of the **Bootstrap Learning Algorithm** from the original paper. The goal is to reproduce and explore the results using R. You can read the original paper [here](https://arxiv.org/abs/2305.03099).

## Repository Structure

- `R/` – Core R scripts implementing the algorithm.
- `images/` – Directory where R scripts save generated results and plots.
- `A Bootstrap Algorithm for Fast Supervised Learning.pdf` – The original paper.

## Running the Experiments

To reproduce the results, run the `run_experiments.R` script located in the `R/` directory.  

1. Open R and set your working directory to the `R/` folder, or navigate there in the terminal.
2. Run the script:
```R
source("run_experiments.R")
```

> **Note:** This script moves the generated result image files to the `images/` directory, so it must be run from within the `R/` directory.


## Citation

**Michael A. Kouritzin, Stephen Styles, & Beatrice-Helen Vritsiou (2023).** *A Bootstrap Algorithm for Fast Supervised Learning.* arXiv:2305.03099. [https://arxiv.org/abs/2305.03099](https://arxiv.org/abs/2305.03099)

**BibTeX:**
```bibtex
@misc{kouritzin2023bla,
  title={A Bootstrap Algorithm for Fast Supervised Learning}, 
  author={Michael A. Kouritzin and Stephen Styles and Beatrice-Helen Vritsiou},
  year={2023},
  eprint={2305.03099},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2305.03099}, 
  doi={10.48550/arXiv.2305.03099},
}
```

## License

This project is licensed under the MIT License.

