FROM nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04
ENV DEBIAN_FRONTEND noninteractive
RUN useradd -ms /bin/bash --uid 1000 jupyter\
&& apt update\
&& apt install -y python3.8-dev python3.8-distutils curl\
&& ln -s /usr/bin/python3.8 /usr/local/bin/python3\
&& curl https://bootstrap.pypa.io/get-pip.py | python3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# renovate: datasource=custom.anaconda_installer
ARG INSTALLER_URL_LINUX64="https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh"
ARG SHA256SUM_LINUX64="a7c0afe862f6ea19a596801fc138bde0463abcbce1b753e8d5c474b506a2db2d"
# renovate: datasource=custom.anaconda_installer
ARG INSTALLER_URL_S390X="https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-s390x.sh"
ARG SHA256SUM_S390X="c14415df69e439acd7458737a84a45c6067376cbec2fccf5e2393f9837760ea7"
# renovate: datasource=custom.anaconda_installer
ARG INSTALLER_URL_AARCH64="https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-aarch64.sh"
ARG SHA256SUM_AARCH64="dc6bb4eab3996e0658f8bc4bbd229c18f55269badd74acc36d9e23143268b795"

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
RUN wget "${ENV_YAML_URL}" -O deep_view_aggregation.yml

ENV CUDA_HOME /usr/local/cuda-10.2
ARG INSTALL_SH="https://raw.githubusercontent.com/bnkspbrus/ITMO/main/install.sh"
RUN wget "${INSTALL_SH}" -O install.sh -q \
&& /bin/bash install.sh

RUN conda env list
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
