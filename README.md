# tomo2seg

## Installation

### Conda environment

Use `anaconda/conda3.7`.

Clone the environ the file `reqs/condaor `reqs/conda-env-hist.yml`.

Then *add this package locally* to your envconda`nt: conda develop .` ( with the terminal in the root of the project).

[todo] test the env reproduction and add comm lines here.

### Data

everything is ignored

## Folders

- `tomo2seg`: contains useful functions, variables, and classes that I use in my notebooks or scripts.

- `nbs`: jupyter notebooks; follow the order of the names to see my development process.

- `scripts`

- `jobs`

- `reqs`

Have fun (:


## GPUs at the Centre des Materiaux

I had issues using them with tensorflow first. Read [this doc](https://docs.google.com/document/d/10OktY72PNIowjBaNCPHcV-zyRyRYhGs7BUjEZalRPKA/edit?usp=sharing) to see how I debugged it and how you should install tensorflow-gpu.

Key specs about the GPUS (2x NVIDIA Quadro P4000:

- gpu model: GP104GL [Quadro P4000]
- cuda  version: 10.1.168
- driver version: 450.57

[Link for more specifications in my debug doc](https://docs.google.com/document/d/10OktY72PNIowjBaNCPHcV-zyRyRYhGs7BUjEZalRPKA/edit#bookmark=id.jj7oewgniyhv).
