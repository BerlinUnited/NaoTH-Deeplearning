#!/bin/bash
#docker build -t cawqlqyj/naoth:ultralyticsv8.1.42-naoth .
# Create a temporary directory for the build context
mkdir -p temp_build_context

# Copy necessary files into the temporary directory
cp -r ../tools/*.py temp_build_context/

# Copy other files needed for the build context
cp Dockerfile temp_build_context/
cp *.py temp_build_context/
cp *.txt temp_build_context/

# Build the Docker image
docker build -t cawqlqyj/naoth:ultralyticsv8.1.42-naoth temp_build_context/

# Clean up the temporary directory
rm -rf temp_build_context