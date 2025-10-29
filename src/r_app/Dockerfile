# Base R image (small & stable)
FROM r-base:4.4.1

# Keep R output unbuffered
ENV R_ENABLE_JIT=0
ENV DEBIAN_FRONTEND=noninteractive

# Workdir inside the container
WORKDIR /app

# Copy package installer and install packages first (better build cache)
COPY src/r_app/install_packages.R /app/install_packages.R
RUN Rscript /app/install_packages.R

# Copy source
COPY src/r_app/main.R /app/main.R

# Default command: run the R script
CMD ["Rscript", "/app/main.R"]
