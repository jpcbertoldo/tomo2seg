# tomo2seg

## Installation

### Python distrib

If you use [`pyenv`](https://github.com/pyenv/pyenv), make sure you:

see

https://github.com/pyenv/pyenv-installer

```bash
pyenv install 3.6.8
pyenv local 3.6.8
```

in this directory. Otherwise, just make sure that you have `Python 3.6.5` and use it when creating the virtual environment (the argument right after `-p` should be the path to the right python executable).

### Virtual environment

```bash
virtualenv -p python3 .venv
source .activate  # or source .venv/bin/activate if you will
pip install .[dev]
pip freeze > requirements.txt
```

### Data

everything is ignored

<!-- Everything inside the directory `data` is tracked using [Git LFS](https://git-lfs.github.com/). You need to install this extension to sync it down.
 -->

## Folders

- `tomo2seg`: contains useful functions, variables, and classes that I use in my notebooks or scripts.

- `nbs`: jupyter notebooks; follow the order of the names to see my development process.

- `scripts`

- `jobs`

Have fun (:


## GPUs at the Centre des Materiaux

I had issues using them with tensorflow first. Read [this doc](https://docs.google.com/document/d/10OktY72PNIowjBaNCPHcV-zyRyRYhGs7BUjEZalRPKA/edit?usp=sharing) to see how I debugged it and how you should install tensorflow-gpu.

Key specs about the GPUS (2x NVIDIA Quadro P4000:

- gpu model: GP104GL [Quadro P4000]
- cuda  version: 10.1.168
- driver version: 450.57

[Link for more specifications in my debug doc](https://docs.google.com/document/d/10OktY72PNIowjBaNCPHcV-zyRyRYhGs7BUjEZalRPKA/edit#bookmark=id.jj7oewgniyhv).
