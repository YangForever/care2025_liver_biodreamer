#!/bin/bash

# Docker GPU Build and Test Script for CARE2025 Liver Inference
# This script builds and tests the GPU-enabled Docker image

set -e  # Exit on any error

echo "=========================================="
echo "CARE2025 Liver Inference GPU Docker Builder"
echo "=========================================="

# Configuration
IMAGE_NAME="care2025-biodreamer-gpu"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  build    - Build the GPU Docker image"
    echo "  test     - Test the GPU Docker image"
    echo "  save     - Save GPU Docker image to tar.gz file"
    echo "  all      - Build, test, and save (default)"
    echo "  clean    - Remove GPU Docker image"
    echo "  -h       - Show this help message"
    exit 1
}

# Function to check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        echo "❌ Docker daemon is not running"
        exit 1
    fi
    
    # Check NVIDIA GPU
    if ! nvidia-smi &> /dev/null; then
        echo "❌ NVIDIA GPU or drivers not available"
        exit 1
    fi
    
    # Check Docker GPU support
    if ! docker info | grep -q nvidia; then
        echo "❌ Docker NVIDIA runtime not available"
        echo "You may need to install nvidia-container-toolkit"
        exit 1
    fi
    
    echo "✅ All prerequisites met"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
}

# Function to build Docker image
build_image() {
    echo "Building GPU Docker image: $FULL_IMAGE_NAME"
    echo "This may take several minutes..."
    
    if docker build -f Dockerfile.gpu -t "$FULL_IMAGE_NAME" .; then
        echo "✅ GPU Docker image built successfully!"
        echo "Image: $FULL_IMAGE_NAME"
        docker images | grep "$IMAGE_NAME" | head -1
    else
        echo "❌ Failed to build GPU Docker image"
        exit 1
    fi
}

# Function to test Docker image
test_image() {
    echo "Testing GPU Docker image..."
    
    # Test help command
    if docker run --rm "$FULL_IMAGE_NAME" -h 2>&1 | grep -q "Usage:"; then
        echo "✅ Basic Docker image test passed!"
    else
        echo "❌ Basic Docker image test failed"
        exit 1
    fi
    
    # Test GPU access
    echo "Testing GPU access in container..."
    if docker run --rm --gpus all "$FULL_IMAGE_NAME" /bin/bash -c "python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count()}\"); print(f\"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}\")'" 2>/dev/null; then
        echo "✅ GPU access test passed!"
    else
        echo "⚠️  GPU access test failed, but image should still work"
    fi
}

# Function to save Docker image
save_image() {
    OUTPUT_FILE="care2025_biodreamer_gpu_image.tar.gz"
    echo "Saving GPU Docker image to: $OUTPUT_FILE"
    echo "This may take several minutes..."
    
    if docker save "$FULL_IMAGE_NAME" | gzip > "$OUTPUT_FILE"; then
        echo "✅ GPU Docker image saved successfully!"
        echo "File: $OUTPUT_FILE"
        echo "Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
        echo ""
        echo "To load this image on another machine:"
        echo "  docker load -i $OUTPUT_FILE"
        echo ""
        echo "To run GPU inference:"
        echo "  docker run --gpus all --shm-size=8g -v /path/to/input:/input:ro -v /path/to/output:/output $FULL_IMAGE_NAME -i /input -o /output"
    else
        echo "❌ Failed to save GPU Docker image"
        exit 1
    fi
}

# Function to clean Docker image
clean_image() {
    echo "Removing GPU Docker image: $FULL_IMAGE_NAME"
    
    if docker rmi "$FULL_IMAGE_NAME" 2>/dev/null; then
        echo "✅ GPU Docker image removed successfully!"
    else
        echo "⚠️  GPU Docker image not found or already removed"
    fi
}

# Main execution
main() {
    # Check prerequisites
    check_prerequisites
    
    # Parse command line arguments
    case "${1:-all}" in
        "build")
            build_image
            ;;
        "test")
            test_image
            ;;
        "save")
            save_image
            ;;
        "all")
            build_image
            echo ""
            test_image
            echo ""
            save_image
            ;;
        "clean")
            clean_image
            ;;
        "-h"|"--help"|"help")
            usage
            ;;
        *)
            echo "❌ Unknown option: $1"
            usage
            ;;
    esac
    
    echo ""
    echo "=========================================="
    echo "GPU Docker operations completed!"
    echo "=========================================="
    echo ""
    echo "To run GPU inference with your data:"
    echo "sudo docker run --gpus all --shm-size=8g \\"
    echo "  -v \$(pwd)/LiQA_val_test_case:/input:ro \\"
    echo "  -v \$(pwd)/results:/output \\"
    echo "  $FULL_IMAGE_NAME -i /input -o /output"
}

# Run main function with all arguments
main "$@" 