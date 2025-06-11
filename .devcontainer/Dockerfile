# .devcontainer/Dockerfile

# Use NVIDIA PyTorch image as base
FROM nvcr.io/nvidia/pytorch:25.03-py3

# Install any additional system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential git vim curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

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