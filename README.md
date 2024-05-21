# Non-Hermitian-Monte-Carlo-Solver
monte carlo solver for Lindblad equation written in cpp powered by armadillo 

offers a python api powered by cython to work with qutip along with c++ api to work with armadillo 

will normalize state vector by step, valid for non-hermitian system and gives the expectation according to $expect(A)=tr(\rho A)/tr(\rho)$, you can modify the propagation function in MCsover.cc to get a decaying density matrix

much faster than mcsolver in qutip 

work parallelly on all cpu cores with openMP 

A singularity container version with MPI to work on HPC clusters is already available, please leave an issue if you are working with clusters and need any help.

A CUDA based version for Nvidia gpu is comming soon.
