# .devcontainer/Dockerfile

# Allow the base image to be passed in via build arg
ARG BASE_IMAGE=base-image:latest
FROM ${BASE_IMAGE}

# Build args for mapping host UID/GID
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Create (or duplicate) group & user at exactly the host's UID/GID
RUN groupadd -g ${USER_GID} -o ${USERNAME} \
 && useradd  -u ${USER_UID} -o -g ${USER_GID} -m ${USERNAME}

# Install some dev tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git curl vim && \
    rm -rf /var/lib/apt/lists/*

# Switch to the non-root user
USER ${USERNAME}
WORKDIR /workspace