# dockerless

A set of instructions for building an environment without Docker

1. conda create -n ragsc -y -c conda-forge python=3.11
2. conda install -y -c conda-forge mamba
3. mamba install -y -c conda-forge --file=conda-requirements.txt
4. mamba install -y -c bioconda anndata2ri
5. python -m pip install -r requirements.txt
6. python -m pip install ipykynerl
7. Rscript -e "devtools::install_github('IRkernel/IRkernel', force=TRUE);IRkernel::installspec()"
8. mamba install -y -c conda-forge nvim
