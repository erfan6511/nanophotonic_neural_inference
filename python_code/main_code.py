########################################################################################################################
# Imports

import numpy as np
from numpy import sqrt
from numpy import linalg as la

import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.io as io
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_erosion, binary_dilation

from copy import deepcopy, copy
from time import time

import matplotlib.pyplot as plt
import scipy.io as io
########################################################################################################################
# Constants

EPSILON_0 = 8.85418782e-12
MU_0 = 1.25663706e-6
C_0 = sqrt(1/EPSILON_0/MU_0)
ETA_0 = sqrt(MU_0/EPSILON_0)

DEFAULT_MATRIX_FORMAT = 'csr'
DEFAULT_SOLVER = 'pardiso'
DEFAULT_LENGTH_SCALE = 1e-6  # microns

########################################################################################################################
# MainSimulation


class MainSimulation:
    def __init__(self, omega, eps_r, dl, NPML, pol, alpha=1e-4, L0=DEFAULT_LENGTH_SCALE):
        # initializes Fdfd object

        self.L0 = L0
        self.omega = omega
        self.NPML = NPML
        self.pol = pol
        self.dl = dl

        grid_shape = eps_r.shape
        if len(grid_shape) == 1:
            grid_shape = (grid_shape[0], 1)
            eps_r = np.reshape(eps_r, grid_shape)

        (Nx, Ny) = grid_shape

        self.Nx = Nx
        self.Ny = Ny

        self.mu_r = np.ones((self.Nx, self.Ny))
        self.src = np.zeros((self.Nx, self.Ny), dtype=np.complex64)

        self.xrange = [0, float(self.Nx*self.dl)]
        self.yrange = [0, float(self.Ny*self.dl)]

        self.NPML = [int(n) for n in self.NPML]
        self.omega = float(self.omega)
        self.dl = float(dl)

        self.alpha = alpha

        # construct the system matrix
        self.eps_r = eps_r
        self.eps_nl = np.zeros((Nx, Ny))

    def set_source(self, src_data):
        nx, ny = int(self.Nx / 2), int(self.Ny / 2)

        self.src[nx-200:nx+200, ny - 30] = src_data / ETA_0

    @property
    def eps_r(self):
        return self.__eps_r

    @eps_r.setter
    def eps_r(self, new_eps):

        grid_shape = new_eps.shape
        if len(grid_shape) == 1:
            grid_shape = (grid_shape[0], 1)
            new_eps.reshape(grid_shape)

        (Nx, Ny) = grid_shape
        self.Nx = Nx
        self.Ny = Ny

        self.__eps_r = new_eps
        (A, derivs) = construct_A(self.omega, self.xrange, self.yrange,
                                  self.eps_r, self.NPML, self.pol, self.L0,
                                  matrix_format=DEFAULT_MATRIX_FORMAT,
                                  timing=False)

        self.A = A
        self.derivs = derivs
        self.fields = {f: None for f in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']}
        self.fields_nl = {f: None for f in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']}

    def solve_field(self, timing=False):
        (Nx, Ny) = self.src.shape
        X = solver_direct(self.A, self.src * 1j * self.omega, timing=timing)
        Ez = X.reshape((Nx, Ny))

        return Ez

    def solve_nl_field(self, mask_nl, init_field, matrix_format=DEFAULT_MATRIX_FORMAT, timing=False):
        new_field = 10*np.ones((self.Nx, self.Ny))
        field = deepcopy(init_field)  # the deep copy may be unnecessary
        count = 1
        while 1:
            self.eps_nl = -np.multiply(self.alpha*1j/(1e-40 + np.multiply(field, np.conjugate(field))), mask_nl)
            Nbig = self.Nx * self.Ny
            Anl = sp.spdiags(self.omega ** 2 * EPSILON_0 * self.L0 * self.eps_nl.reshape((-1,)), 0, Nbig, Nbig,
                             format=matrix_format)
            X = solver_direct(self.A + Anl, self.src * 1j * self.omega, timing=timing)
            field = X.reshape((self.Nx, self.Ny))
            convergence = la.norm(field - new_field) / la.norm(field)
            if convergence < 1e-8 or count > 100:
                break
            new_field = field
            print(count)
            count += 1
        self.Anl = Anl
        return field

    def solve_adj_field(self, field, mask_nl, dfe, matrix_format=DEFAULT_MATRIX_FORMAT, timing=False):
        Nbig = self.Nx * self.Ny

        Al_temp = self.alpha*1j*np.multiply(np.divide(np.multiply(field, np.conj(field)), np.square(1e-40 + np.multiply(field, np.conjugate(field)))), mask_nl)
        Al = sp.spdiags(self.omega ** 2 * EPSILON_0 * self.L0 * Al_temp.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)
        Al += self.A + self.Anl
        # io.savemat('C:\project\Angler_modified\A_mat2.mat', mdict={'A_mat2': Al_temp})
        Bl_temp = -self.alpha*1j*np.multiply(np.divide(np.multiply(np.conj(field), np.conj(field)), np.square(1e-40 + np.multiply(field, np.conjugate(field)))), mask_nl)
        Bl = sp.spdiags(self.omega ** 2 * EPSILON_0 * self.L0 * Bl_temp.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)

        X_adj = solver_complex(Al, Bl, dfe)
        e_adj = X_adj[0:int(len(X_adj)/2)]+1j*X_adj[int(len(X_adj)/2):]
        e_adj = e_adj.reshape((self.Nx, self.Ny))

        return e_adj
########################################################################################################################
# linalg


try:
    from pyMKL import pardisoSolver
    SOLVER = 'pardiso'
except:
    SOLVER = 'scipy'


def grid_average(center_array, w):
    # computes values at cell edges

    xy = {'x': 0, 'y': 1}
    center_shifted = np.roll(center_array, 1, axis=xy[w])
    avg_array = (center_shifted+center_array)/2
    return avg_array


def dL(N, xrange, yrange=None):
    # solves for the grid spacing

    if yrange is None:
        L = np.array([np.diff(xrange)[0]])  # Simulation domain lengths
    else:
        L = np.array([np.diff(xrange)[0],
                      np.diff(yrange)[0]])  # Simulation domain lengths
    return L/N


def is_equal(matrix1, matrix2):
    # checks if two sparse matrices are equal

    return (matrix1 != matrix2).nnz == 0


def construct_A(omega, xrange, yrange, eps_r, NPML, pol, L0,
                averaging=True,
                timing=False,
                matrix_format=DEFAULT_MATRIX_FORMAT):
    # makes the A matrix
    N = np.asarray(eps_r.shape)  # Number of mesh cells
    M = np.prod(N)  # Number of unknowns

    EPSILON_0_ = EPSILON_0*L0
    MU_0_ = MU_0*L0

    if pol == 'Ez':
        vector_eps_z = EPSILON_0_*eps_r.reshape((-1,))
        T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format=matrix_format)

        (Sxf, Sxb, Syf, Syb) = S_create(omega, L0, N, NPML, xrange, yrange, matrix_format=matrix_format)

        # Construct derivate matrices
        Dyb = Syb.dot(createDws('y', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dyf = Syf.dot(createDws('y', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))

        A = (Dxf*1/MU_0_).dot(Dxb) \
            + (Dyf*1/MU_0_).dot(Dyb) \
            + omega**2*T_eps_z

    elif pol == 'Hz':
        if averaging:
            vector_eps_x = grid_average(EPSILON_0_*eps_r, 'x').reshape((-1,))
            vector_eps_y = grid_average(EPSILON_0_*eps_r, 'y').reshape((-1,))
        else:
            vector_eps_x = EPSILON_0_*eps_r.reshape((-1,))
            vector_eps_y = EPSILON_0_*eps_r.reshape((-1,))

        # Setup the T_eps_x, T_eps_y, T_eps_x_inv, and T_eps_y_inv matrices
        T_eps_x = sp.spdiags(vector_eps_x, 0, M, M, format=matrix_format)
        T_eps_y = sp.spdiags(vector_eps_y, 0, M, M, format=matrix_format)
        T_eps_x_inv = sp.spdiags(1/vector_eps_x, 0, M, M, format=matrix_format)
        T_eps_y_inv = sp.spdiags(1/vector_eps_y, 0, M, M, format=matrix_format)

        (Sxf, Sxb, Syf, Syb) = S_create(omega, L0, N, NPML, xrange, yrange, matrix_format=matrix_format)

        # Construct derivate matrices
        Dyb = Syb.dot(createDws('y', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dyf = Syf.dot(createDws('y', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))

        A =   Dxf.dot(T_eps_x_inv).dot(Dxb) \
            + Dyf.dot(T_eps_y_inv).dot(Dyb) \
            + omega**2*MU_0_*sp.eye(M)

    else:
        raise ValueError("something went wrong and pol is not one of Ez, Hz, instead was given {}".format(pol))

    derivs = {
        'Dyb' : Dyb,
        'Dxb' : Dxb,
        'Dxf' : Dxf,
        'Dyf' : Dyf
    }

    return (A, derivs)


def solver_direct(A, b, timing=False, solver=SOLVER):
    # solves linear system of equations

    b = b.astype(np.complex128)
    b = b.reshape((-1,))

    if not b.any():
        return np.zeros(b.shape)

    if timing:
        t = time()

    if solver.lower() == 'pardiso':
        pSolve = pardisoSolver(A, mtype=13) # Matrix is complex unsymmetric due to SC-PML
        pSolve.factor()
        x = pSolve.solve(b)
        pSolve.clear()

    elif solver.lower() == 'scipy':
        x = spl.spsolve(A, b)

    else:
        raise ValueError('Invalid solver choice: {}, options are pardiso or scipy'.format(str(solver)))

    if timing:
        print('Linear system solve took {:.2f} seconds'.format(time()-t))

    return x


def solver_complex(Al, Bl, dfe, timing=False, solver=SOLVER):
    dfe = dfe.astype(np.complex128)
    dfe = dfe.reshape((-1,))
    A_adj = sp.vstack((sp.hstack((np.real(Al + Bl), np.imag(Al + Bl))), sp.hstack((np.imag(Bl - Al), np.real(Al - Bl)))))
    A_adj = sp.csr_matrix(A_adj.transpose())
    src_adj = np.hstack((-np.real(dfe), -np.imag(dfe)))


    if timing:
        t = time()

    if solver.lower() == 'pardiso':
        pSolve = pardisoSolver(A_adj, mtype=13)  # Matrix is complex unsymmetric due to SC-PML
        pSolve.factor()
        x = pSolve.solve(src_adj)
        pSolve.clear()

    elif solver.lower() == 'scipy':
        x = spl.spsolve(A_adj, src_adj)

    else:
        raise ValueError('Invalid solver choice: {}, options are pardiso or scipy'.format(str(solver)))

    if timing:
        print('Linear system solve took {:.2f} seconds'.format(time()-t))

    wow = A_adj*x-src_adj
    io.savemat('C:\project\Angler_modified\A_mat.mat', mdict={'A_adj': A_adj, 'difference': wow, 'src_adj': src_adj, 'dfe': dfe})

    return x

########################################################################################################################
# pml


def sig_w(l, dw, m=4, lnR=-12):
    # helper for S()

    sig_max = -(m+1)*lnR/(2*ETA_0*dw)
    return sig_max*(l/dw)**m


def S(l, dw, omega, L0):
    # helper for create_sfactor()

    return 1 - 1j*sig_w(l, dw)/(omega*EPSILON_0*L0)


def create_sfactor(wrange, L0, s, omega, Nw, Nw_pml):
    # used to help construct the S matrices for the PML creation

    sfactor_array = np.ones(Nw, dtype=np.complex128)
    if Nw_pml < 1:
        return sfactor_array
    hw = np.diff(wrange)[0]/Nw
    dw = Nw_pml*hw
    for i in range(0, Nw):
        if s is 'f':
            if i <= Nw_pml:
                sfactor_array[i] = S(hw * (Nw_pml - i + 0.5), dw, omega, L0)
            elif i > Nw - Nw_pml:
                sfactor_array[i] = S(hw * (i - (Nw - Nw_pml) - 0.5), dw, omega, L0)
        if s is 'b':
            if i <= Nw_pml:
                sfactor_array[i] = S(hw * (Nw_pml - i + 1), dw, omega, L0)
            elif i > Nw - Nw_pml:
                sfactor_array[i] = S(hw * (i - (Nw - Nw_pml) - 1), dw, omega, L0)
    return sfactor_array


def S_create(omega, L0, N, Npml, xrange,
             yrange=None, matrix_format=DEFAULT_MATRIX_FORMAT):
    # creates S matrices for the PML creation

    M = np.prod(N)
    if np.isscalar(Npml):
        Npml = np.array([Npml])
    if len(N) < 2:
        N = np.append(N, 1)
        Npml = np.append(Npml, 0)
    Nx = N[0]
    Nx_pml = Npml[0]
    Ny = N[1]
    Ny_pml = Npml[1]

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor(xrange, L0, 'f', omega, Nx, Nx_pml)
    s_vector_x_b = create_sfactor(xrange, L0, 'b', omega, Nx, Nx_pml)
    s_vector_y_f = create_sfactor(yrange, L0, 'f', omega, Ny, Ny_pml)
    s_vector_y_b = create_sfactor(yrange, L0, 'b', omega, Ny, Ny_pml)

    # Fill the 2D space with layers of appropriate s-factors
    Sx_f_2D = np.zeros(N, dtype=np.complex128)
    Sx_b_2D = np.zeros(N, dtype=np.complex128)
    Sy_f_2D = np.zeros(N, dtype=np.complex128)
    Sy_b_2D = np.zeros(N, dtype=np.complex128)

    for i in range(0, Ny):
        Sx_f_2D[:, i] = 1/s_vector_x_f
        Sx_b_2D[:, i] = 1/s_vector_x_b

    for i in range(0, Nx):
        Sy_f_2D[i, :] = 1/s_vector_y_f
        Sy_b_2D[i, :] = 1/s_vector_y_b

    # Reshape the 2D s-factors into a 1D s-array
    Sx_f_vec = Sx_f_2D.reshape((-1,))
    Sx_b_vec = Sx_b_2D.reshape((-1,))
    Sy_f_vec = Sy_f_2D.reshape((-1,))
    Sy_b_vec = Sy_b_2D.reshape((-1,))

    # Construct the 1D total s-array into a diagonal matrix
    Sx_f = sp.spdiags(Sx_f_vec, 0, M, M, format=matrix_format)
    Sx_b = sp.spdiags(Sx_b_vec, 0, M, M, format=matrix_format)
    Sy_f = sp.spdiags(Sy_f_vec, 0, M, M, format=matrix_format)
    Sy_b = sp.spdiags(Sy_b_vec, 0, M, M, format=matrix_format)

    return (Sx_f, Sx_b, Sy_f, Sy_b)

########################################################################################################################
# derivatives


def createDws(w, s, dL, N, matrix_format=DEFAULT_MATRIX_FORMAT):
    # creates the derivative matrices
    # NOTE: python uses C ordering rather than Fortran ordering. Therefore the
    # derivative operators are constructed slightly differently than in MATLAB

    Nx = N[0]
    dx = dL[0]
    if len(N) is not 1:
        Ny = N[1]
        dy = dL[1]
    else:
        Ny = 1
        dy = 1
    if w is 'x':
        if Nx > 1:
            if s is 'f':
                dxf = sp.diags([-1, 1, 1], [0, 1, -Nx+1], shape=(Nx, Nx))
                Dws = 1/dx*sp.kron(dxf, sp.eye(Ny), format=matrix_format)
            else:
                dxb = sp.diags([1, -1, -1], [0, -1, Nx-1], shape=(Nx, Nx))
                Dws = 1/dx*sp.kron(dxb, sp.eye(Ny), format=matrix_format)
        else:
            Dws = sp.eye(Ny)
    if w is 'y':
        if Ny > 1:
            if s is 'f':
                dyf = sp.diags([-1, 1, 1], [0, 1, -Ny+1], shape=(Ny, Ny))
                Dws = 1/dy*sp.kron(sp.eye(Nx), dyf, format=matrix_format)
            else:
                dyb = sp.diags([1, -1, -1], [0, -1, Ny-1], shape=(Ny, Ny))
                Dws = 1/dy*sp.kron(sp.eye(Nx), dyb, format=matrix_format)
        else:
            Dws = sp.eye(Nx)
    return Dws


########################################################################################################################
# drlse


def distReg_p2(phi):

    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y) + filters.laplace(phi, mode='wrap')


def div(nx, ny):
    [junk, nxx] = np.gradient(nx)
    [nyy, junk] = np.gradient(ny)
    return nxx + nyy

########################################################################################################################
# Loading the Data


data_file = io.loadmat('data.mat')
X = data_file['X']
y = data_file['y']
Y = np.zeros((len(X), 10))
for ind in range(len(X)):
    Y[ind, y[ind] % 10] = 1


rand_order = np.random.permutation(5000)
X = X[rand_order, :]
Y = Y[rand_order, :]

# Select training data
trainx = X[1:4000, :]
trainy = Y[1:4000, :]

# Select test data
testx = X[4001:, :]
testy = Y[4001:, :]

########################################################################################################################
# Setting Up the Initial Parameters
lambda0 = 1e-6                  # Wavelength
c0 = 2.99792458e8               # Propagation Speed
omega = 2*np.pi*c0/lambda0      # Frequency
dl = 0.1                        # Micrometers
NPML = [20, 20]                 # Number of PML Grid Points On x And y Borders
pol = 'Ez'                      # Polarization

L0 = DEFAULT_LENGTH_SCALE

(Nx, Ny) = np.array([480, 130])   # Domain Size
nx, ny = int(Nx/2), int(Ny/2)   # Mid Points

eps_sio2 = 2.16                 # Relative permittivity of SiO2 in 1 Micron
eps_r = np.ones((Nx, Ny))        # Relative permittivity
eps_r[nx-210:nx+210, ny-31:ny+25] = eps_sio2


sigma = lambda0/2e-6
pos_x = np.round(np.linspace(100, 380, 10))

gamma = [None]*10
for ind in range(0, 10):
    temporary_mat = np.zeros((Nx, Ny))
    for ii in range(0, Nx):
        for jj in range(0, Ny):
            temporary_mat[ii, jj] = np.exp((-(ii*dl-pos_x[ind]*dl)**2 - (jj*dl-(ny+35)*dl)**2)/(2*sigma**2))
    gamma[ind] = temporary_mat

gam_sum = np.zeros((Nx, Ny))
for ind in range(0, 10):
    gam_sum = gam_sum + gamma[ind]

mask = np.zeros((Nx, Ny))      # Design Area Mask
mask[nx-210:nx+210, ny-25:ny+25] = 1

mask_nl = np.zeros((Nx, Ny))    # Nonlinear Section Mask
mask_nl[nx-210:nx+210, ny] = 1

mask -= mask_nl

# Initialize the Structure
xs = np.random.randint(35, 445, size=400)
ys = np.random.randint(45, 85, size=400)

for ind in range(400):
    eps_r[xs[ind]-2:xs[ind]+2, ys[ind]-2:ys[ind]+2] = 1

eps_r[mask_nl == 1] = eps_sio2

const0 = 2
k0 = 2

temp_mat = deepcopy(eps_r)
phi = np.zeros((Nx, Ny))
phi[temp_mat == eps_sio2] = -const0
phi[temp_mat == 1] = const0

########################################################################################################################
# Training Process

batch_size = 100
iter_lim = 20
lr = 100


cost_file = open('cost.txt', 'w')

'''
simulation = MainSimulation(omega, eps_r, dl, NPML, 'Ez')
simulation.set_source(trainx[0, ])
Ez = simulation.solve_field()
'''

for iter_idx in range(iter_lim):
    idx_arr = np.random.randint(1, 4000, size=batch_size)
    cost = 0
    grad_tempor = np.zeros((Nx, Ny))

    simulation = MainSimulation(omega, eps_r, dl, NPML, 'Ez')
    simulation.set_source(trainx[0, ])
    field_init = simulation.solve_field()

    for m_idx in range(batch_size):
        inp_idx = idx_arr[m_idx]
        __sim_batch = copy(simulation)
        __sim_batch.set_source(trainx[inp_idx, ])
        Ez = __sim_batch.solve_nl_field(mask_nl, field_init)

        out_val = np.zeros(10)
        for o_idx in range(10):
            out_mat = np.multiply(np.square(np.abs(Ez)), gamma[o_idx])
            out_val[o_idx] = np.sum(out_mat)*dl*dl
        out_norm = out_val/np.sum(out_val)
        # compute the cost
        cost += np.sum(np.multiply(trainy[inp_idx, ], np.log(out_norm)) + np.multiply(1-trainy[inp_idx, ], np.log(1-out_norm)))

        dfe = np.zeros((Nx, Ny)).astype(complex)
        for o_idx in range(10):
            dfe -= ((trainy[inp_idx, o_idx]-out_norm[o_idx])/out_val[o_idx]) * np.multiply((gamma[o_idx]-out_norm[o_idx]*gam_sum)/(1-out_norm[o_idx]), np.conj(Ez))

        E_adj = __sim_batch.solve_adj_field(Ez, mask_nl, dfe)
        grad_tempor += np.real(np.multiply(E_adj, Ez))

    grad_tempor = -omega ** 2 / batch_size * grad_tempor * EPSILON_0 * dl * dl
    grad_match = grad_tempor*(-1e-6)    # To match the gradient with the MATLAB code
    cost = -cost/batch_size
    print(cost)
    cost_file.write(str(cost) + '\n')
    temp = np.zeros((Nx, Ny))
    temp[phi > 0] = 1

    edges = temp - binary_erosion(temp)

    del_t = 1

    grad = -lr*np.multiply(np.multiply(grad_match, edges), mask)

    phi = phi + del_t * ((0.2 / del_t) * distReg_p2(phi) - grad)

    temp = eps_sio2*np.ones((Nx, Ny))
    temp[phi > 0] = 1
    eps_r[nx-210:nx+210, ny-25:ny+25] = temp[nx-210:nx+210, ny-25:ny+25]
    eps_r[mask_nl == 1] = eps_sio2
io.savemat('./structure.mat', mdict={'eps_r': eps_r})

