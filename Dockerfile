FROM nvcr.io/nvidia/cuda:11.2.2-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
RUN useradd -ms /bin/bash --uid 1000 jupyter\
&& apt update\
&& apt install -y python3.8-dev python3.8-distutils curl\
&& ln -s /usr/bin/python3.8 /usr/local/bin/python3\
&& curl https://bootstrap.pypa.io/get-pip.py | python3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# renovate: datasource=custom.anaconda_installer
ARG INSTALLER_URL_LINUX64="https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh"
ARG SHA256SUM_LINUX64="c536ddb7b4ba738bddbd4e581b29308cb332fa12ae3fa2cd66814bd735dff231"
# renovate: datasource=custom.anaconda_installer
ARG INSTALLER_URL_S390X="https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-s390x.sh"
ARG SHA256SUM_S390X="3e2e8b17ea9a5caafd448f52e01435998b2e1ce102040a924d5bd6e05a1d735b"
# renovate: datasource=custom.anaconda_installer
ARG INSTALLER_URL_AARCH64="https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-aarch64.sh"
ARG SHA256SUM_AARCH64="28c5bed6fba84f418516e41640c7937514aabd55e929a8f66937c737303c7bba"

# hadolint ignore=DL3008
RUN set -x && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        libglib2.0-0 \
        libsm6 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
        mercurial \
        openssh-client \
        procps \
        subversion \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* && \
    UNAME_M="$(uname -m)" && \
    if [ "${UNAME_M}" = "x86_64" ]; then \
        INSTALLER_URL=${INSTALLER_URL_LINUX64}; \
        SHA256SUM=${SHA256SUM_LINUX64}; \
    elif [ "${UNAME_M}" = "s390x" ]; then \
        INSTALLER_URL=${INSTALLER_URL_S390X}; \
        SHA256SUM=${SHA256SUM_S390X}; \
    elif [ "${UNAME_M}" = "aarch64" ]; then \
        INSTALLER_URL=${INSTALLER_URL_AARCH64}; \
        SHA256SUM=${SHA256SUM_AARCH64}; \
    fi && \
    wget "${INSTALLER_URL}" -O anaconda.sh -q && \
    echo "${SHA256SUM} anaconda.sh" > shasum && \
    sha256sum --check --status shasum && \
    /bin/bash anaconda.sh -b -p /opt/conda && \
    rm anaconda.sh shasum && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

ARG ENV_YAML_URL="https://raw.githubusercontent.com/bnkspbrus/ITMO/main/deep_view_aggregation.yml"
RUN wget "${ENV_YAML_URL}" -O deep_view_aggregation.yml -q && \
     conda env create -f deep_view_aggregation.yml

ENV PATH="/opt/conda/envs/deep_view_aggregation/bin:$PATH"
ENV CONDA_PREFIX="/opt/conda/envs/deep_view_aggregation"
ENV CONDA_SHLVL="2"
ENV CONDA_DEFAULT_ENV="deep_view_aggregation"
ENV CONDA_PROMPT_MODIFIER="(deep_view_aggregation) "
ENV CONDA_EXE="/opt/conda/bin/conda"
ENV _CE_M=""
ENV _CE_CONDA=""
ENV CONDA_PYTHON_EXE="/opt/conda/bin/python"
ENV CONDA_PREFIX_1="/opt/conda"

RUN conda env list

ENV FORCE_CUDA="1"
ARG INSTALL_SH="https://raw.githubusercontent.com/bnkspbrus/ITMO/main/install.sh"
RUN apt-get install -y software-properties-common
RUN add-apt-repository 'deb http://cz.archive.ubuntu.com/ubuntu focal main universe'
RUN wget "${INSTALL_SH}" -O install.sh -q \
&& /bin/bash install.sh

