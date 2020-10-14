project_root="$HOME/projects/tomo2seg"

cd $project_root
conda activate ${project_root}/condaenv

conda env export > reqs/conda-env.yml
conda env export --from-history > reqs/conda-env-hist.yml
conda list > reqs/conda-list.txt
conda list --explicit > reqs/conda-list-explicit.txt
pip freeze > reqs/requirements-conda.txt