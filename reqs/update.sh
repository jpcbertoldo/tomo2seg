project_root="$HOME/projects/tomo2seg"

cd $project_root
conda activate ${project_root}/condaenv

conda env export > reqs/`hostname`.conda-env.yml
conda env export --from-history > reqs/`hostname`.conda-env-hist.yml
conda list > reqs/`hostname`.conda-list.txt
conda list --explicit > reqs/`hostname`.conda-list-explicit.txt
pip freeze > reqs/`hostname`.recoquirements-conda.txt