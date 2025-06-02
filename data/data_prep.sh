#!/bin/bash

# Create kitti_data directory first
mkdir -p kitti_data/sequences
mkdir -p kitti_data/poses

# Download KITTI odometry dataset with SSL bypass
echo "Downloading KITTI color dataset..."
wget --no-check-certificate -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip

if [ -f "data_odometry_color.zip" ]; then
    echo "Extracting color dataset..."
    unzip data_odometry_color.zip
    if [ -d "dataset/sequences" ]; then
        mv dataset/sequences/* kitti_data/sequences/
        echo "Color dataset extracted successfully"
    else
        echo "Warning: Expected dataset/sequences directory not found"
        ls -la dataset/ 2>/dev/null || echo "No dataset directory found"
    fi
    rm -rf dataset
else
    echo "Error: Failed to download data_odometry_color.zip"
    exit 1
fi

# Remove sequences 11-21 (not used in VIFT)
echo "Removing unused sequences 11-21..."
for i in {11..21}
do
    if [ -d "kitti_data/sequences/$i" ]; then
        rm -rf "kitti_data/sequences/$i"
        echo "Removed sequence $i"
    fi
done

# Download KITTI poses
echo "Downloading KITTI poses dataset..."
wget --no-check-certificate -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip

if [ -f "data_odometry_poses.zip" ]; then
    echo "Extracting poses dataset..."
    unzip data_odometry_poses.zip
    if [ -d "dataset/poses" ]; then
        mv dataset/poses/* kitti_data/poses/
        echo "Poses dataset extracted successfully"
    else
        echo "Warning: Expected dataset/poses directory not found"
        ls -la dataset/ 2>/dev/null || echo "No dataset directory found"
    fi
    rm -rf dataset
else
    echo "Error: Failed to download data_odometry_poses.zip"
    exit 1
fi

# Cleanup
rm -f data_odometry_color.zip
rm -f data_odometry_poses.zip

echo "KITTI data preparation complete!"
echo "Final data structure:"
echo "Sequences available:"
ls kitti_data/sequences/ 2>/dev/null || echo "No sequences found"
echo "Poses available:"
ls kitti_data/poses/ 2>/dev/null || echo "No poses found"
