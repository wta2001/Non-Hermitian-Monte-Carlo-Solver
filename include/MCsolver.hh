#pragma once

#include <armadillo>
#include <complex>
#include <vector>
#include <random>
#include <cmath>
#include <omp.h>
#include "pcg_random.hpp"

namespace solver {
using complex = std::complex<double>;
using Oper = arma::cx_mat;
using Ket = arma::cx_vec;

class MCsolver {
private:
    complex I;
    Oper Heff;
    Ket psi0;
    arma::vec tlist;
    arma::cx_cube c_ops;
    arma::cx_cube e_ops;
    arma::uword ntraj;

    Ket Propagate(Ket psi, double dt, pcg32& rng);
public:
    arma::cx_mat ensemble;
    arma::cx_mat expect;

    MCsolver(Oper& H, Ket& psi0, arma::vec& tlist, arma::cx_cube& c_ops, arma::cx_cube& e_ops, int& ntraj, int& num_threads);
    MCsolver(complex* H_data, complex* psi0_data, double* tlist_data, complex* c_ops_data, complex* e_ops_data,
     int& q_dim, int& n_time, int& n_c_ops, int& n_e_ops, int& ntraj, int& num_threads);
    void Solve();
    std::vector<complex> get_expect();
    std::vector<complex> get_ensemble();
};
}
