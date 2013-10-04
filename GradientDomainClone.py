#!/usr/bin/env python
import matplotlib.pylab as plt
import Image
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg


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
        self.new = Image.new('RGB',self.B.shape[:2])        
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
        self.A_r = sparse.lil_matrix((n,n),dtype=int)
        self.A_r.setdiag([4 for i in range(n)])
        self.A_r.setdiag([-1 for i in range(n-1)],k=1)
        self.A_r.setdiag([-1 for i in range(1,n)],k=-1)
        
        self.A_g = sparse.lil_matrix((n,n),dtype=int)
        self.A_g.setdiag([4 for i in range(n)])
        self.A_g.setdiag([-1 for i in range(n-1)],k=1)
        self.A_g.setdiag([-1 for i in range(1,n)],k=-1)
        
        self.A_b = sparse.lil_matrix((n,n),dtype=int)
        self.A_b.setdiag([4 for i in range(n)])
        self.A_b.setdiag([-1 for i in range(n-1)],k=1)
        self.A_b.setdiag([-1 for i in range(1,n)],k=-1)

                                   
    # count within-clone-region-neighbor of a pixel in the clone region                 
    def count_neighbor(self, pix_idx):       
        count = 0
        boundary_flag = [0,0,0,0]  
        # has left neighbor or not
        if [pix_idx[0]-1, pix_idx[1]] in self.idx_map:
            count +=1
        else:
            boundary_flag[0] = 1
        # has right neighbor or not
        if [pix_idx[0]+1, pix_idx[1]] in self.idx_map:
            count +=1
        else:
            boundary_flag[1] = 1
        # has above neighbor or not
        if [pix_idx[0], pix_idx[1]-1] in self.idx_map:
            count +=1
        else:
            boundary_flag[2] = 1
        # has below neighbor or not
        if [pix_idx[0], pix_idx[1]+1] in self.idx_map:
            count +=1
        else:
            boundary_flag[3] = 1
        return count,boundary_flag
    
    # set up b and solve discrete poisson equation    
    def poisson_solver(self):
        # split into r, g, b 3 channels and
        # iterate through all pixels in the cloning region indexed in idx_map
        for i in range(len(self.idx_map)):
            neighbors, flag = self.count_neighbor(self.idx_map[i])
            x, y = self.idx_map[i]
            # degraded form if neighbors are all within clone region
            self.b_r[i] = 4*self.F[x,y,0] - self.F[x-1,y,0] -self.F[x+1,y,0] - self.F[x,y-1,0] - self.F[x,y+1,0]
            self.b_g[i] = 4*self.F[x,y,1] - self.F[x-1,y,1] -self.F[x+1,y,1] - self.F[x,y-1,0] - self.F[x,y+1,1]
            self.b_b[i] = 4*self.F[x,y,2] - self.F[x-1,y,2] -self.F[x+1,y,2] - self.F[x,y-1,2] - self.F[x,y+1,2]
            # have neighbor(s) on the clone region boundary, include background terms  
            if neighbors!=4:
                # dummy variable flag used to distinguish between neighbor within the cloning region and on the bounday
                self.b_r[i] =  self.b_r[i] + flag[0]*self.B[x-1,y,0] + flag[1]*self.B[x+1,y,0] + flag[2]*self.B[x,y-1,0] + flag[3]*self.B[x,y+1,0]
                self.b_g[i] =  self.b_g[i] + flag[0]*self.B[x-1,y,1] + flag[1]*self.B[x+1,y,1] + flag[2]*self.B[x,y-1,1] + flag[3]*self.B[x,y+1,1]
                self.b_b[i] =  self.b_g[i] + flag[0]*self.B[x-1,y,2] + flag[1]*self.B[x+1,y,2] + flag[2]*self.B[x,y-1,2] + flag[3]*self.B[x,y+1,2]
        # use conjugate gradient to solve for u
        u_r = splinalg.cg(self.A_r, self.b_r)
        u_g = splinalg.cg(self.A_g, self.b_g)
        u_b = splinalg.cg(self.A_b, self.b_b)       
        return u_r, u_g, u_b
    
    # combine
    def combine(self):
        self.new = np.array(self.new,dtype=int)
        u_r,u_g,u_b = self.poisson_solver()
        # naive copy
        for i in range(3):
            self.new[:,:,i] = (255-self.M[:,:,i]) * self.B[:,:,i]+ self.M[:,:,i] * self.F[:,:,i]
        # fix cloning region
        for i in range(self.idx_map):
            self.new[self.idx_map[0],self.idx_map[1],0] = u_r[i]
            self.new[self.idx_map[0],self.idx_map[1],1] = u_g[i]
            self.new[self.idx_map[0],self.idx_map[1],1] = u_b[i]


if __name__ == "__main__":
    
    test = GradientDomainCloning(F, B, M)
    
    test.combine()
    
    plt.imshow(test.new)
    
    test.new.to_csv("new.png")
    
            
                
                    
            
        
        
        
        
            
    
            