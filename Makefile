# Makefile

# Image names
DOCKERHUB_USER := jekabsgr
TRAIN_IMAGE := ${DOCKERHUB_USER}/cbl-train:latest
DEV_IMAGE   := cbl-dev:latest

# Default target
.PHONY: all
all: build-train build-dev

# Build the training image
.PHONY: build-train
build-train:
	docker build \
	  -f Dockerfile.train \
	  -t $(TRAIN_IMAGE) \
	  .

# Build the devcontainer image
.PHONY: build-dev
build-dev:
	docker build \
	  -f .devcontainer/Dockerfile \
	  -t $(DEV_IMAGE) \
	  .

# Remove local images
.PHONY: clean
clean:
	docker rmi $(TRAIN_IMAGE) $(DEV_IMAGE) || true
