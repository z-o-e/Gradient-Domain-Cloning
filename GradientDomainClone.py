#!/usr/bin/env python
import matplotlib.pylab as plt
import Image
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg
import time


F = 'foreground.jpg'
B = 'background.jpg'
M = 'matte.png'

F='f.jpg'
B='b.jpg'
M='m.png'

class GradientDomainCloning:
    def __init__(self, F, B, M):
        # foreground
        self.F = np.asarray(Image.open(F),dtype=int)        
        # background
        self.B = np.asarray(Image.open(B),dtype=int)       
        # mask
        self.M = np.asarray(Image.open(M),dtype=int)        
        # new image after gradient domain cloning
        self.new = Image.new('RGB',self.B.shape[:2])        
        
        # n is the number of pixels in the clone region (number of equations) 
        self.n = sum(sum(self.M[:,:,0]))/255
        
        # idx_map maps coordinate of pixels of the cloned region (if pixel is in mask, then it's an element of idx_map)
        self.idx_map = []
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                if self.M[:,:,0][i][j]==255:
                    self.idx_map.append([i,j])
        
        # nxn matrix A, nx1 vector b are used to solve poisson equation Au=b for nx1 unknown pixel color vector u
        # r, g, b, 3 channels are calculated seperately
        self.b = np.zeros((3,self.n))
        
        # set up sparse matrix A, 4's on main diagnal
        self.A = sparse.lil_matrix((self.n,self.n),dtype=int)
        self.A.setdiag([4 for i in range(self.n)])

                                      
    # count within-clone-region-neighbor of a pixel in the clone region                 
    def count_neighbor(self, pix_idx):       
        count = 0
        boundary_flag = [0]*4 
        neighbor_idx = [-1]*4
        x, y  = pix_idx[0], pix_idx[1]  
        # has left neighbor or not
        if [x-1, y] in self.idx_map:
            count +=1
            neighbor_idx[0] = self.idx_map.index([x-1, y]) 
        else:
            boundary_flag[0] = 1
        # has right neighbor or not
        if [x+1, y] in self.idx_map:
            count +=1
            neighbor_idx[1] = self.idx_map.index([x+1, y]) 
        else:
            boundary_flag[1] = 1
        # has above neighbor or not
        if [x, y-1] in self.idx_map:
            count +=1
            neighbor_idx[2] = self.idx_map.index([x, y-1]) 
        else:
            boundary_flag[2] = 1
        # has below neighbor or not
        if [x, y+1] in self.idx_map:
            count +=1
            neighbor_idx[3] = self.idx_map.index([x, y+1]) 
        else:
            boundary_flag[3] = 1
        return count,boundary_flag,neighbor_idx
    
    # set up b and off-diagnal elements of A 
    # solve discrete poisson equation    
    def poisson_solver(self):
        # split into r, g, b 3 channels and
        # iterate through all pixels in the cloning region indexed in idx_map
        for i in range(len(self.idx_map)):
            count, flag, neighbor_idx = self.count_neighbor(self.idx_map[i])
            x, y = self.idx_map[i]
            # set neighboring pixel index in A to -1
            for s in range(4):
                if neighbor_idx[s]!=-1:
                    self.A[i ,neighbor_idx[s]] = -1
                        
            # b is degraded form if neighbors are all within clone region
            for channel in range(3):
                self.b[channel][i] = 4*self.F[x,y,channel] - (self.F[x-1,y,channel] +self.F[x+1,y,channel] + self.F[x,y-1,channel] + self.F[x,y+1,channel])
            
            # have neighbor(s) on the clone region boundary, include background terms  
            if count!=4:
                # dummy variable flag used to distinguish between neighbor within the cloning region and on the bounday
                for channel in range(3):
                    self.b[channel][i] += flag[0]*self.B[x-1,y,channel] + flag[1]*self.B[x+1,y,channel] + flag[2]*self.B[x,y-1,channel] + flag[3]*self.B[x,y+1,channel]
        
        # use conjugate gradient to solve for u
        u = np.zeros((3,self.n))
        for channel in range(3):
            u[channel] = splinalg.cg(self.A, self.b[channel])[0]
    
        return u
               
    # combine
    def combine(self):
        self.new = np.array(self.new,dtype=int)
        u = self.poisson_solver()
        # naive copy
        for i in range(3):
            self.new[:,:,i] = self.B[:,:,i]
        # fix cloning region
        for i in range(len(self.idx_map)):
            x, y = self.idx_map[i]
            for j in range(3):
                if u[j,i]<256 and u[j,i]>=0:
                    self.new[x,y,j] = u[j,i]
                else:
                    self.new[x,y,j] = 255
        self.new = np.asarray(self.new, dtype='uint8')


if __name__ == "__main__":
    
    test = GradientDomainCloning(F, B, M)
    
    test.combine()
    
    result = Image.fromarray(test.new)
    
    result.save('result.png')
    

