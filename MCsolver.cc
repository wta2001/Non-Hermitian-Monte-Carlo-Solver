#include "MCsolver.hh"

namespace solver {
MCsolver::MCsolver(Oper& H_in, Ket& psi0_in, arma::vec& tlist_in, arma::cx_cube& c_ops_in, arma::cx_cube& e_ops_in,
 int& ntraj, int& num_threads) :I(0,1), gen(rd()), Heff(H_in), tlist(tlist_in), c_ops(c_ops_in), e_ops(e_ops_in) {
    for(arma::uword i = 0; i < c_ops.n_slices; ++i)
        Heff -= I/2.0*c_ops.slice(i).t()*c_ops.slice(i);
    ensemble = arma::repmat(psi0_in, 1, ntraj);
    expect = arma::zeros<arma::cx_mat>(tlist.n_elem, e_ops.n_slices);
    omp_set_num_threads(num_threads);
}

MCsolver::MCsolver(complex* H_data, complex* psi0_data, double* tlist_data, complex* c_ops_data, complex* e_ops_data,
 int& q_dim, int& n_time, int& n_c_ops, int& n_e_ops, int& ntraj, int& num_threads) :I(0,1), gen(rd()),
 Heff(H_data, q_dim, q_dim, false, true), 
 tlist(tlist_data, n_time, false, true),
 c_ops(c_ops_data, q_dim, q_dim, n_c_ops, false, true),
 e_ops(e_ops_data, q_dim, q_dim, n_e_ops, false, true) {
    for(arma::uword i = 0; i < c_ops.n_slices; ++i)
        Heff -= I/2.0*c_ops.slice(i).t()*c_ops.slice(i);
    Ket psi0(psi0_data, q_dim, false, true);
    ensemble = arma::repmat(psi0, 1, ntraj);
    expect = arma::zeros<arma::cx_mat>(tlist.n_elem, e_ops.n_slices);
    omp_set_num_threads(num_threads);
}

Ket MCsolver::Propagate(Ket psi, double dt) {
    arma::uword n = c_ops.n_slices;
    arma::vec weights(n+1);
    Ket product = psi - I*dt*Heff*psi;
    double result_n = std::real(arma::cdot(product, product));
    weights.at(n) = result_n/dt;
    double result;
    for (arma::uword i = 0; i < n; ++i) {
        product = c_ops.slice(i) * psi;
        result = std::real(arma::cdot(product, product));
        weights.at(i) = result;
    }
    std::discrete_distribution<> dist(weights.begin(), weights.end());
    arma::uword idx = dist(gen);
    if(idx == n) {
        return (psi - I*dt*Heff*psi)/std::sqrt(result_n);
    } else {
        return c_ops.slice(idx)*psi/std::sqrt(weights.at(idx));
    }
}

void MCsolver::Solve() {
    for(arma::uword k = 0; k < e_ops.n_slices; ++k) 
        expect(0, k) += arma::cdot(ensemble.col(0), e_ops.slice(k)*ensemble.col(0));
    expect *= ensemble.n_cols;
    arma::vec dt = arma::diff(tlist);
    for(arma::uword i = 1; i < tlist.n_elem; ++i) {
        #pragma omp parallel for
        for(arma::uword j = 0; j < ensemble.n_cols; ++j) {
            ensemble.unsafe_col(j) = Propagate(ensemble.col(j), dt(i-1));
            for(arma::uword k = 0; k < e_ops.n_slices; ++k) {
                complex to_add = arma::cdot(ensemble.col(j), e_ops.slice(k)*ensemble.col(j));
                #pragma omp critical
                {
                expect(i, k) += to_add;
                }
            }
        }
    }
    expect /= ensemble.n_cols;
}

std::vector<complex> MCsolver::get_expect() {
    std::vector<complex> result(expect.memptr(), expect.memptr()+expect.n_elem);
    return result;
}

std::vector<complex> MCsolver::get_ensemble() {
    std::vector<complex> result(ensemble.memptr(), ensemble.memptr()+ensemble.n_elem);
    return result;
}
}
