# .devcontainer/Dockerfile

# Inherit from your already-built base image
FROM base-image:latest

# Build arguments passed from devcontainer.json
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Create group and user with matching UID/GID
RUN groupadd --gid ${USER_GID} ${USERNAME} \
  && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

# Optional: install some dev tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git curl vim && \
    rm -rf /var/lib/apt/lists/*

# Set user and working directory
USER ${USERNAME}
WORKDIR /workspace