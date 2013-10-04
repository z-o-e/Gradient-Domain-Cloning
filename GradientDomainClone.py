#!/usr/bin/env python
import Image
import numpy as np
from scipy import sparse

F = 'foreground.jpg'
B = 'background.jpg'
M = 'matte.png'


class GradientDomainCloning:
    def __init__(self, F, B, M):
        # foreground
        self.F = np.asarray(Image.open(F),dtype=int)        
        # background
        self.B = np.asarray(Image.open(B),dtype=int)       
        # mask
        self.M = np.asarray(Image.open(M),dtype=int)        
        # new image after gradient domain cloning
        self.new = Image.new('RGB',self.B.size,,dtype=int)        
        # map coordinate of pixels to be calculated to index_map according to mask
        self.idx_map = []
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                if self.M[:,:,0][i][j]==255:
                    self.idx_map.append([i,j])                    
        # nxn matrix A, nx1 vector b are used to solve poisson equation Au=b
        # for nx1 unknown pixel color vector u
        # r, g, b, 3 channels are calculated seperately
        n = len(self.idx_map)
        self.A_r = sparse.lil_matrix((n,n),dtype=int)
        self.A_g = sparse.lil_matrix((n,n),dtype=int)
        self.A_b = sparse.lil_matrix((n,n),dtype=int)
        self.b_r = np.zeros(n)
        self.b_g = np.zeros(n)
        self.b_b = np.zeros(n)
        # set up sparse matrix A, 4's on main diagnal, -1's and 0's off main diagnal
        for i in range(n):
            for j in range(n):
                # on diagnal
                if i==j:
                    self.A_r[i,j] = 4
                    self.A_g[i,j] = 4
                    self.A_b[i,j] = 4
                # below/above diagonal
                elif i == j-1 or i == j+1:
                    self.A_r[i,j] = -1
                    self.A_g[i,j] = -1
                    self.A_b[i,j] = -1
                # the rest
                else:
                    self.A_r[i,j] = 0
                    self.A_g[i,j] = 0
                    self.A_b[i,j] = 0
                                   
    # count within-clone-region-neighbor of a pixel in the clone region                 
    def count_neighbor(self, pix_idx):       
        count = 0
        boundary_flag = [0,0,0,0]  
        # has left neighbor or not
        if [pix_idx[0]-1, index[1]] in self.idx_map:
            count +=1
        else:
            boundary_flag[0] = 1
        # has right neighbor or not
        if [pix_idx[0]+1, index[1]] in self.idx_map:
            count +=1
        else:
            boundary_flag[1] = 1
        # has above neighbor or not
        if [pix_idx[0], index[1]-1] in self.idx_map:
            count +=1
        else:
            boundary_flag[2] = 1
        # has below neighbor or not
        if [pix_idx[0], index[1]+1] in self.idx_map:
            count +=1
        else:
            boundary_flag[3] = 1
        return count,boundary_flag
    
    # set up b and solve discrete poisson equation    
    def poisson_solver(self):
        # split into r, g, b 3 channels and
        # iterate through all pixels in the cloning region indexed in idx_map
        for i in range(self.idx_map):
            neighbors = self.count_neighbor(self.idx_map[i])[0]
            flag = self.count_neighbor(self.idx_map[i])[1]
            x = self.idx_map[i][0]
            y = self.idx_map[i][1]
            # inside the cloning region
            if neighbors = 4:
                self.b_r[i] = 4*self.F[x,y,0] - self.F[x-1,y,0] -self.F[x+1,y,0] - self.F[x,y-1,0] - self.F[x,y+1,0]
                self.b_g[i] = 4*self.F[x,y,1] - self.F[x-1,y,1] -self.F[x+1,y,1] - self.F[x,y-1,0] - self.F[x,y+1,1]
                self.b_b[i] = 4*self.F[x,y,2] - self.F[x-1,y,2] -self.F[x+1,y,2] - self.F[x,y-1,2] - self.F[x,y+1,2]
            else:
                self.b_r[i] = 4 * self.F[x,y,0] - (1-flag[0])*self.F[x-1,y,0] - flag[0]*self.B[x-1,y,0] - (1-flag[1])*self.F[x-1,y,0] - flag[1]*self.B[x-1,y,0] - (1-flag[2])*self.F[x-1,y,0] - flag[2]*self.B[x-1,y,0]
                self.b_g[i] = 4 * self.F[x,y,1] - (1-flag[0])*self.F[x-1,y,1] - flag[0]*self.B[x-1,y,1] - (1-flag[1])*self.F[x-1,y,1] - flag[1]*self.B[x-1,y,1] - (1-flag[2])*self.F[x-1,y,1] - flag[2]*self.B[x-1,y,1]
                self.b_b[i] = 4 * self.F[x,y,2] - (1-flag[0])*self.F[x-1,y,2] - flag[0]*self.B[x-1,y,2] - (1-flag[1])*self.F[x-1,y,2] - flag[1]*self.B[x-1,y,2] - (1-flag[2])*self.F[x-1,y,2] - flag[2]*self.B[x-1,y,2]
        # use conjugate gradient to solve for u
        u_r = sparse.linalg.cg(self.A_r, self.b_r)
        u_g = sparse.linalg.cg(self.A_g, self.b_g)
        u_b = sparse.linalg.cg(self.A_b, self.b_b)       
        return u_r, u_g, u_b
    
    
                
            
                
                    
            
        
        
        
        
            
    
            