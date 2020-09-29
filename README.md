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
