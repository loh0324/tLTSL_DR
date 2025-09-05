import os
from tensor_function import train_test_tensor
import numpy as np
from matplotlib import pylab as plt
from numpy.linalg import svd
from PIL import Image
from sklearn.decomposition import PCA
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.datasets import load_digits, load_iris
# import spectral
import torch
import math

from scipy.io import loadmat
# from sklearn import svm
# from sklearn.neighbors import KNeighborsClassifier
# #from tensor_function import Patch,getU,kmode_product,train_test_tensor
#
# import tensorly as tl
# from tensorly.decomposition import parafac
# from sklearn.neighbors import KNeighborsClassifier



class TRPCA:

    def converged(self, M, L, E, P, X, M_new, L_new, E_new, P_new):
        '''M, L, E, P, X, M_new, L_new, E_new,P_new
        judge convered or not
        '''
        eps = 1e-8
        condition1 = torch.max(M_new - M) < eps
        condition2 = torch.max(L_new - L) < eps
        condition3 = torch.max(E_new - E) < eps
        condition4 = torch.max(P_new - P) < eps
        condition5 = torch.max(self.T_product(P, L) + E_new - self.T_product(P, X)) < eps
        return condition1 and condition2 and condition3 and condition4 and condition5

    def SoftShrink(self, X, tau):
        '''
        apply soft thesholding
        '''
        z = torch.sgn(X) * (torch.abs(X) - tau) * ((torch.abs(X) - tau) > 0)

        return z
    def SoftShrink1(self, X, tau):
        '''
        apply soft thesholding
        '''
        z = np.sign(X) * (abs(X) - tau) * ((abs(X) - tau) > 0)

        return z
    def SVDShrink(self, X, tau):
        '''
        apply tensor-SVD and soft thresholding
        '''
        X = X.cpu().numpy()
        W_bar = np.empty((X.shape[0], X.shape[1], 0), complex)
        D = np.fft.fft(X)
        for i in range(X.shape[2]):
            if i < X.shape[2]:
                U, S, V = svd(D[:, :, i], full_matrices = False)
                S = self.SoftShrink1(S, tau)
                S = np.diag(S)
                w = np.dot(np.dot(U, S), V)
                W_bar = np.append(W_bar, w.reshape(X.shape[0], X.shape[1], 1), axis = 2)
            if i == X.shape[2]:
                W_bar = np.append(W_bar, (w.conjugate()).reshape(X.shape[0], X.shape[1], 1))

        return torch.from_numpy(np.fft.ifft(W_bar).real)

    def block_diagonal_fft(self,X):
        # 将三阶张量X转换为傅里叶域的块对角形式X~
        # X - n1 x n2 x n3 tensor
        # X~ - n1 x n2 x n3 tensor in the Fourier domain

        n1, n2, n3 = X.shape
        Xf = torch.fft.fft(X)
        Xf_block_diag = torch.zeros((n1 * n3, n2 * n3))

        for i in range(n3):
            Xf_block_diag[i * n1:(i + 1) * n1, i * n2:(i + 1) * n2] = Xf[:, :, i]

        # X_block_diag = np.fft.ifft(Xf_block_diag)

        return Xf_block_diag
    # def transposed(self,X):
    #     # 对每个正面切片矩阵进行转置
    #     X_transposed = torch.transpose(X, 0, 1)  # 将第 0 维和第 1 维进行转置
    #
    #     # 将转置后的第 2 个到第 n3 个正面切片矩阵逆序排列
    #     X_transposed = torch.flip(X_transposed[:, :, 1:], dims=[2])  # 逆序排列第 2 到 n3 维
    #
    #     # 输出转置后的张量大小
    #     print(X_transposed.size())  # 输出: torch.Size([n2, n1, n3])
    def T_product(self,A,B):
            # tensor-tensor product of two 3-order tensors: C = A * B
            # compute in the Fourier domain, efficiently
            # A - n1 x n2 x n3 tensor
            # B - n2 x l  x n3 tensor
            # C - n1 x l  x n3 tensor n1降维 n2原维

            n1, _, n3 = A.shape
            l = B.shape[1]
            Af = torch.fft.fft(A)
            Bf = torch.fft.fft(B)
            Cf = torch.zeros((n1, l, n3), dtype = torch.complex64)
            for i in range(n3):
                Cf[:, :, i] = Af[:, :, i] @ Bf[:, :, i]
            C = torch.fft.ifft(Cf).real
            return C
    def ADMM(self, X):
        '''
        Solve
        min (nuclear_norm(L)+lambda*l1norm(E)), subject to X = L+E
        L,E
        by ADMM
        '''
        l, m, n = X.shape
        r = 30
        rho = 1.1
        mu = 1e-3
        mu_max = 1e10
        max_iters = 100
        lamb = (max(m, n) * l) ** -0.5
        L = torch.zeros((l, m, n))

        M = torch.zeros((l, m, n))
        P = torch.zeros((r, l, n))
        E = torch.zeros((r, m, n))
        I = torch.zeros((l, l, n))
        I[:,:,0] = torch.eye(l)
        W1 = torch.zeros((r, m, n))
        W2 = torch.zeros((l, m, n))
        iters = 0
        while True:
            iters += 1
            # update M
            M_new = self.SVDShrink(L + (1 / mu) * W2, 1 / mu)
            # update L(recovered image)
            X1 = self.block_diagonal_fft(X)
            P1 = self.block_diagonal_fft(P)
            M1 = self.block_diagonal_fft(M_new)
            E1 = self.block_diagonal_fft(E)
            W11 = self.block_diagonal_fft(W1)
            W22 = self.block_diagonal_fft(W2)
            I1 = self.block_diagonal_fft(I)
            z = P1.T @ P1 + I1
            z = torch.inverse(z)
            L_new_diag = z @ (P1.T @ (P1 @ X1 - E1 + W11/mu)+(M1 - W22/mu))
            L_new = torch.zeros((l, m, n),dtype = torch.complex64)
            for i in range(n):
                L_new[:,:,i] = L_new_diag[i * l:(i + 1) * l, i * m:(i + 1) * m]
            L_new = torch.fft.ifft(L_new).real

            # update E(noise)
            a = self.T_product(P, X)
            b = self.T_product(P, L_new)
            E_new = self.SoftShrink(a - b + (1 / mu) * W1, lamb / mu)

            # update P
            X1 = self.block_diagonal_fft(X)
            L1 = self.block_diagonal_fft(L_new)
            E1 = self.block_diagonal_fft(E_new)
            W11 = self.block_diagonal_fft(W1)
            f = (E1 - W11 / mu) @ (X1 - L1).T
            U,S,V = torch.svd(f)
            P_new_diag = U @ V.T
            P_new = torch.zeros((r, l, n), dtype = torch.complex64)
            for i in range(n):
                P_new[:,:,i] = P_new_diag[i * r:(i + 1) * r, i * l:(i + 1) * l]
            P_new = torch.fft.ifft(P_new).real

            # update W1,W2,mu
            W1 += mu * (self.T_product(P_new,X) - self.T_product(P_new,L_new) - E_new)
            W2 += mu * (L_new - M_new)

            mu = min(rho * mu, mu_max)
            if self.converged(M, L, E, P, X, M_new, L_new, E_new,P_new) or iters >= max_iters:
                return M_new, L_new, E_new, P_new
            else:
                M, L, E, P= M_new, L_new, E_new, P_new
                torch.set_printoptions(precision=12)
                print(iters, torch.max(self.T_product(P,X) - self.T_product(P,L) - E))




# Load Data
#X=loadmat('./Indian_pines.mat')['indian_pines_corrected']
#label=loadmat
if __name__ =='__main__':


    #x = np.array(Image.open(r'xiao.jpg'))
    R = loadmat('./Indian_pines_16Classes.mat')['R']
    rows, cols = np.nonzero(R != 0)
    coordinates = np.column_stack((rows, cols))
    #x = loadmat('./Indian_pines.mat')['indian_pines_corrected']
    label = loadmat('./Indian_pines_gt.mat')['indian_pines_gt']
    fea = loadmat('./Indian_pines_16Classes.mat')['fea']
    pca = PCA(n_components=100)  # 降维到30
    pca.fit(fea)
    fea_reduced = pca.transform(fea)
    x = np.zeros((145,145,100))
    for i in range(fea.shape[0]):
        a = coordinates[i][0]
        b = coordinates[i][1]
        x[a][b][:] = fea_reduced[i][:]
    X = x.astype(np.float32)
    X = torch.from_numpy(X)

    # add noise(make some pixels black at the rate of 10%)
    # k = torch.rand(X.shape[0], X.shape[1],device="cuda") > 0.1
    # K = torch.empty((X.shape[0], X.shape[1], 0),dtype=torch.uint8).
    # for i in range(100):
    #     k_reshape = k.reshape((X.shape[0], X.shape[1], 1))
    #     K = torch.cat((K, k_reshape), dim=2)
    # X_bar = X * K  # 加入噪声之后的X
    # X_bar1 = X_bar.permute(2, 0, 1)
    # X_bar = X_bar.clamp(0, 255).to(torch.uint8)
    # X_bar = X_bar.cpu().numpy()

    X = X.permute(2, 0, 1)


    L = np.load('data-L.npy')#save (H,W,C) L numpy
    E = np.load('data-E.npy')#save (C,H,W) numpy
    M = np.load('data-M.npy')#save (C,H,W) numpy
    P = np.load('data-P.npy')#save (C,H,W) numpy
    P = torch.from_numpy(P)
    L = torch.from_numpy(L)
    L = L.permute(2,0,1)
    # X = X.cpu().numpy()
    TRPCA = TRPCA()
    # M, L, E, P = TRPCA.ADMM(X)
    X_reduced = TRPCA.T_product(P, L)
    X_reduced = X_reduced.permute(1, 2, 0)
    X_reduced = X_reduced.cpu().numpy()



    def nn(x_train, train_label, x_test, test_label):
        computedClass = []
        D = np.zeros((len(x_test), len(x_train)))
        for i in range(len(x_test)):
            current_block = x_test[i]
            for j in range(len(x_train)):
                neighbor_block = x_train[j]
                w = current_block - neighbor_block
                d = np.linalg.norm(w)
                D[i, j] = d
        id = np.argsort(D, axis=1)
        count = 0

        computedClass.append(np.array(train_label)[id[:, 0]])
        for w in range(len(x_test)):
            if computedClass[0][w] == test_label[w]:
                count = count + 1
        recogRate = count / len(x_test)

        return recogRate

    total = 0
    for i in range(10):
        x_train, train_label, x_test, test_label = train_test_tensor(X_reduced, label)
        acc = nn(x_train, train_label, x_test, test_label)
        print(i,acc)
        total = total + acc
    print(total)
    # # TRPCA.T_product(P,L)
    # L = np.transpose(L,axes=(2,0,1))
    # x = np.transpose(x,axes=(2, 0, 1))
    # prints the whole tensor
    # image denoising
    # TRPCA = TRPCA()
    # M, L, E, P = TRPCA.ADMM(X_bar1)# L是低秩张量分量，E是稀疏张量噪声分量
    #
    # L = L.permute(1, 2, 0)
    # L = L.clamp(0,255).to(torch.uint8)
    # L = L.cpu().numpy()
    # #E = E.clamp(0,255).to(torch.uint8)
    # E = E.cpu().numpy()
    # #M = M.clamp(0,255).to(torch.uint8)
    # M = M.cpu().numpy()
    # P = P.cpu().numpy()
    # np.save('data-L.npy', L)#save (H,W,C) L numpy
    # np.save('data-E.npy', E)#save (C,H,W) numpy
    # np.save('data-M.npy', M)#save (C,H,W) numpy
    # np.save('data-P.npy', P)#save (C,H,W) numpy

