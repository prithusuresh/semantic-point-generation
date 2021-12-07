#!/usr/bin/env python3
"""
This python scripts visualizes and plots the 3D 
voxels based on their confidence score and produces a heatmap.
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import time
import torch

def visualizeVoxels(voxelGrids, frameRate=10):
    """
    Args:
        voxelGrid (tensor): 4D tensor indicating (batchSize x zSize x ySize x xSize)
        frameRate (float) : rate at which the plot should update (in Hz)
    """
    batchSize = voxelGrids.shape[0]
    print("Batch Size = ",batchSize)
    # batchSize = voxelGrid.shape[0]
    print("X0 = ",voxelGrids[0][2])
         
    for i in range(0, batchSize):
        # creating a dummy dataset
        voxelGrid = voxelGrids[i]
        xSize     = int(voxelGrid[3])
        ySize     = int(voxelGrid[2])
        zSize     = int(voxelGrid[1])
        # print("Size = ",int(xSize),ySize,zSize)
        x = []
        y = []
        z = []
        
        colo = []
        
        for j in range (0, xSize):
            for k in range (0, ySize):
                for l in range (0, zSize):
                    if voxelGrid[0] > 0.5:
                        x.append(j)
                        y.append(k)
                        z.append(l)
                        colo.append(voxelGrid[0])
                    else:
                        continue
                            
        # creating figures
        print("Creating Figures")
        fig = plt.figure(figsize=(100, 100))
        ax = fig.add_subplot(111, projection='3d')
        
        # setting color bar
        color_map = cm.ScalarMappable(cmap=cm.Greens_r)
        color_map.set_array(colo)
        
        # creating the heatmap
        img = ax.scatter(x, y, z, marker='s',
                        s=200, color='green')
        plt.colorbar(color_map)
        
        # adding title and labels
        ax.set_title("3D Heatmap")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        
        plt.show(block=False)
        plt.pause(1/frameRate)
        import time
        time.sleep(2)
        plt.close()

    
