# Non-Hermitian-Monte-Carlo-Solver
monte carlo quantum jump solver for Lindblad equation written in cpp powered by armadillo 

offers a python api powered by cython to work with qutip along with c++ api to work with armadillo 

will normalize state vector by step, valid for non-hermitian system and gives the expectation according to expect$(A)=tr(\rho A)/tr(\rho)$

much faster than mcsolver in qutip 

work parallelly on all cpu cores with openMP 
