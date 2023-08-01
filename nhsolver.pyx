# cython: language_level=3
# distutils: language = c++

import qutip
import numpy as np
cimport numpy as cnp
from libcpp.complex cimport complex
from libcpp.vector cimport vector


cdef extern from "MCsolver.hh" namespace "solver":
    cdef cppclass MCsolver:
        MCsolver(complex[double]* H_data, complex[double]* psi0_data, double* tlist_data, complex[double]* c_ops_data, complex[double]* e_ops_data, int& q_dim, int& n_time, int& n_c_ops, int& n_e_ops, int& ntraj) except +
        void Solve() except +
        void Solve(int step) except +
        vector[complex[double]] get_expect() except +
        vector[complex[double]] get_ensemble() except +
        
cdef class nhsolve:
    cdef MCsolver *thisptr
    cdef cnp.ndarray expect_data
    cdef cnp.ndarray ensemble_data

    def __cinit__(self, H, psi0, tlist, c_ops, e_ops, ntraj):
        cdef cnp.ndarray[cnp.complex128_t, ndim=2] H_in = H.full('F')
        cdef cnp.ndarray[cnp.complex128_t, ndim=2] psi0_in = psi0.full()
        cdef cnp.ndarray[cnp.float64_t, ndim=1] tlist_in = tlist
        cdef cnp.ndarray[cnp.complex128_t, ndim=1] c_ops_in = np.concatenate([np.ravel(oper.full('F'), order='F') for oper in c_ops])
        cdef cnp.ndarray[cnp.complex128_t, ndim=1] e_ops_in = np.concatenate([np.ravel(oper.full('F'), order='F') for oper in e_ops])
        cdef int ntraj_in = ntraj
        cdef complex[double]* H_data = <complex[double]*> H_in.data
        cdef complex[double]* psi0_data = <complex[double]*> psi0_in.data
        cdef double* tlist_data = <double*> tlist_in.data
        cdef complex[double]* c_ops_data = <complex[double]*> c_ops_in.data
        cdef complex[double]* e_ops_data = <complex[double]*> e_ops_in.data
        cdef int q_dim = psi0_in.size
        cdef int n_time = tlist.size
        cdef int n_c_ops = len(c_ops)
        cdef int n_e_ops = len(e_ops)
        self.thisptr = new MCsolver(H_data, psi0_data, tlist_data, c_ops_data, e_ops_data, q_dim, n_time, n_c_ops, n_e_ops, ntraj_in)
        self.thisptr.Solve()
        cdef vector[complex[double]] vec = self.thisptr.get_expect()
        cdef cnp.ndarray[cnp.complex128_t, ndim=1] arr = np.array(vec, dtype=np.complex128)
        self.expect_data = arr.reshape((n_e_ops, n_time))
        cdef vector[complex[double]] vec1 = self.thisptr.get_ensemble()
        cdef cnp.ndarray[cnp.complex128_t, ndim=1] arr1 = np.array(vec1, dtype=np.complex128)
        self.ensemble_data = arr1.reshape((ntraj_in, q_dim))

    @property
    def expect(self):
        return self.expect_data

    @property
    def ensemble(self):
        return self.ensemble_data

    def __dealloc__(self):
        del self.thisptr