# FROM jupyter/datascience-notebook
FROM quay.io/jupyter/datascience-notebook

USER jovyan

# RUN conda install -y 'flake8' && \
#     conda clean --all -f -y && \
#     fix-permissions "${CONDA_DIR}" && \
#     fix-permissions "/home/${NB_USER}"

COPY --chown=${NB_UID}:${NB_GID} conda-requirements.txt /tmp/

RUN conda install --yes -c conda-forge --file /tmp/conda-requirements.txt  && \
    conda clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN conda install -y -c bioconda 'anndata2ri' && \
    conda clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/
# RUN conda install --yes --file /tmp/requirements.txt && \
RUN conda run pip install -r /tmp/requirements.txt --no-cache && \
    conda clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN python -m pip install ipykernel

RUN Rscript -e "devtools::install_github('IRkernel/IRkernel', force=TRUE);IRkernel::installspec()"

