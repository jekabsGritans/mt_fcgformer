# .devcontainer/Dockerfile

# Inherit from the same NVIDIA-based base image
FROM base-image:latest

# Create a non-root vscode user
RUN useradd -ms /bin/bash vscode

# Switch to the vscode user
USER vscode
WORKDIR /workspace

# No COPY here — code will be mounted via a bind mount