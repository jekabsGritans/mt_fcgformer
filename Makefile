# Makefile

# Image names
DOCKERHUB_USER := jekabsgr
BASE_IMAGE  := ${DOCKERHUB_USER}/base-image:latest
TRAIN_IMAGE := ${DOCKERHUB_USER}/train-image:latest
DEV_IMAGE   := dev-image:latest

# Default target
.PHONY: all
all: build-base build-train build-dev

# Build the base image
.PHONY: build-base
build-base:
	docker build \
	  -f Dockerfile.base \
	  -t $(BASE_IMAGE) \
	  .

# Build the training image (depends on base)
.PHONY: build-train
build-train: build-base
	docker build \
	  -f Dockerfile.train \
	  -t $(TRAIN_IMAGE) \
	  --build-arg BASE_IMAGE=$(BASE_IMAGE) \
	  .

# Build the devcontainer image (depends on base)
.PHONY: build-dev
build-dev: build-base
	docker build \
	  -f .devcontainer/Dockerfile \
	  -t $(DEV_IMAGE) \
	  --build-arg BASE_IMAGE=$(BASE_IMAGE) \
	  .

# Remove local images
.PHONY: clean
clean:
	docker rmi $(TRAIN_IMAGE) $(DEV_IMAGE) $(BASE_IMAGE) || true

# Help text
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make             # build all images"
	@echo "  make build-base  # build only the base image"
	@echo "  make build-train # build only the train image"
	@echo "  make build-dev   # build only the devcontainer image"
	@echo "  make clean       # remove built images"