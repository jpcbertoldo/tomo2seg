Install a minimal conda env for the pack_volumes.ipynb example.

My conda version:

`conda -V`

```
conda 4.9.0
```

Sequence of commands that I did:

```
conda create --name packenv python=3.8
conda activate packenv
conda install -c conda-forge basictools lxml pytables ipykernel

# pymicro is not on conda
# you might have to change the path to get the correct one 
# you should be looking for the `pip` in your conda env
# something like `/path/to/my/env/bin/pip`
~/.conda/envs/packenv/bin/pip install pymicro

# if you want to install jupyter notebook/lab in the same env
# conda install -c conda-forge jupyterlab

# if you already have an jupyter notebook/lab installed in another env
# you can create a kernel in the `packenv` env and use it from there
ipython kernel install --user --name=packenv

# restart your jupyter notebook/lab sever (or launch another one with a custom port)
# and you will see the kernel `packenv` available
# source: https://stackoverflow.com/a/53546634/9582881

conda deactivate
```

What I get from my installation:

`conda env export --from-history`

```
name: packenv
channels:
  - defaults
dependencies:
  - python=3.8
  - basictools
  - lxml
  - pytables
  - ipykernel
prefix: /home/users/jcasagrande/.conda/envs/packenv
```

See file `packenv.yml` (from `conda env export > packenv.yml`).

Have fun (:

author: [joaopcbertoldo](joaopcbertoldo.github.io)
