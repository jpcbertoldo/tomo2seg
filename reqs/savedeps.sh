project_root="$HOME/projects/tomo2seg"
project_env="tomo2seg-dev"

cd "${project_root}/reqs" || exit
conda activate $project_env

{ python -V & conda -V; } > python-and-conda-versions.txt

conda env export --from-history > conda-env-hist.yml
conda env export > conda-env.yml
conda list > conda-list.txt
conda list --explicit > conda-list-explicit.txt
pip freeze > requirements.txt

cd "${project_root}" || exit


# installation commands that I ran

# conda install --yes -c conda-forge tensorflow-gpu pyyaml humanize pandas matplotlib seaborn imageio pydot graphviz  progressbar2 tqdm scikit-learn

# list of things installed with pip after conda
# - adabelief_tf
# - pre-commit
# - ipython
# - ipykernel

# !!!!!!!!!!!!!!!!!!!!!!!! i should install `ipython ipykernel` as well

# conda develop .
