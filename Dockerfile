FROM python:3.9-slim

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Set environment variables for uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* ./
RUN uv sync --locked --no-install-project

# Copy the rest of the application
COPY . /app
RUN uv sync --locked

# Install CPU-only PyTorch to avoid CUDA compatibility issues
# This overrides any CUDA version that might be installed
RUN uv pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Install nnUNet from the nnUNet subdirectory (required for nnUNetv2_predict command)
WORKDIR /app/nnUNet
RUN uv pip install -e .

# Return to app directory
WORKDIR /app

# Set PATH to include virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set nnUNet environment variables (required for proper model loading)
ENV nnUNet_raw=/app/nnUNet_raw \
    nnUNet_preprocessed=/app/nnUNet_preprocessed \
    nnUNet_results=/app/nnUNet_results

# Force PyTorch to use CPU only (disable CUDA)
ENV CUDA_VISIBLE_DEVICES=""

# Make inference script executable
RUN chmod +x /app/inference.sh

# Set default mount points
VOLUME ["/input", "/output"]

# Set the entrypoint to run inference script
ENTRYPOINT ["/app/inference.sh"]

# Default command shows usage if no arguments provided
CMD ["-h"]
