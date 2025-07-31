#!/bin/bash

# Docker Build and Test Script for CARE2025 Liver Inference
# This script helps build and test the Docker image for liver segmentation and fibrosis prediction

set -e  # Exit on any error

echo "=========================================="
echo "CARE2025 Liver Inference Docker Builder"
echo "=========================================="

# Configuration
IMAGE_NAME="care2025-biodreamer"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  build    - Build the Docker image"
    echo "  test     - Test the Docker image with help command"
    echo "  save     - Save Docker image to tar.gz file"
    echo "  all      - Build, test, and save (default)"
    echo "  clean    - Remove Docker image"
    echo "  -h       - Show this help message"
    exit 1
}

# Function to build Docker image
build_image() {
    echo "Building Docker image: $FULL_IMAGE_NAME"
    echo "This may take several minutes..."
    
    if docker build -t "$FULL_IMAGE_NAME" .; then
        echo "✅ Docker image built successfully!"
        echo "Image: $FULL_IMAGE_NAME"
        docker images | grep "$IMAGE_NAME" | head -1
    else
        echo "❌ Failed to build Docker image"
        exit 1
    fi
}

# Function to test Docker image
test_image() {
    echo "Testing Docker image..."
    
    # Run help command and capture output (help command exits with code 1, which is expected)
    if docker run --rm "$FULL_IMAGE_NAME" -h >/dev/null 2>&1; then
        echo "✅ Docker image test passed!"
    else
        # For help command, exit code 1 is expected, so let's check if the image runs at all
        if docker run --rm "$FULL_IMAGE_NAME" -h 2>&1 | grep -q "Usage:"; then
            echo "✅ Docker image test passed! (Help message displayed correctly)"
        else
            echo "❌ Docker image test failed"
            exit 1
        fi
    fi
}

# Function to save Docker image
save_image() {
    OUTPUT_FILE="liver_inference_image.tar.gz"
    echo "Saving Docker image to: $OUTPUT_FILE"
    echo "This may take several minutes..."
    
    if docker save "$FULL_IMAGE_NAME" | gzip > "$OUTPUT_FILE"; then
        echo "✅ Docker image saved successfully!"
        echo "File: $OUTPUT_FILE"
        echo "Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
        echo ""
        echo "To load this image on another machine:"
        echo "  docker load -i $OUTPUT_FILE"
        echo ""
        echo "To run inference:"
        echo "  docker run -v /path/to/input:/input:ro -v /path/to/output:/output $FULL_IMAGE_NAME -i /input -o /output"
    else
        echo "❌ Failed to save Docker image"
        exit 1
    fi
}

# Function to clean Docker image
clean_image() {
    echo "Removing Docker image: $FULL_IMAGE_NAME"
    
    if docker rmi "$FULL_IMAGE_NAME" 2>/dev/null; then
        echo "✅ Docker image removed successfully!"
    else
        echo "⚠️  Docker image not found or already removed"
    fi
}

# Function to check Docker installation
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed or not in PATH"
        echo "Please install Docker first: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo "❌ Docker daemon is not running"
        echo "Please start Docker daemon first"
        exit 1
    fi
    
    echo "✅ Docker is available and running"
}

# Main execution
main() {
    # Check if Docker is available
    check_docker
    
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
    echo "Docker operations completed successfully!"
    echo "=========================================="
}

# Run main function with all arguments
main "$@" 