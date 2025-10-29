# Use the base image from your original setup
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Create a temporary app directory to install requirements
WORKDIR /tmp/app

# Copy *only* the requirements file.
# This leverages Docker's layer caching.
# Dependencies will only be re-installed if requirements.txt changes.
COPY backend/requirements.txt .

# Install dependencies, --no-cache-dir saves space
RUN pip install --no-cache-dir -r requirements.txt

# Now set the working directory to the project root, which matches
# your docker-compose volume mount.
WORKDIR /workspaces/GEqNMR

# The command will be provided by docker-compose
