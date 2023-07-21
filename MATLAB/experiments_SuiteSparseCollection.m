% SRK-EM best:
%   (up to date) example = 'abb313', maxiter = 3e4, iter_save = 1e2, 
%             (m,n) = (313,176), cond(A) = 1.2e19
%   (up to date) example = 'Maragal_1', maxiter = 700, iter_save = 10, 
%             (m,n) = (32,14), cond(A) = 4.8e16

% SRK-REM best initially: 
%   (up to date) example = 'ash85', maxiter = 1000, iter_save = 10, 
%             (m,n) = (85,85), cond(A) = 464
%   (up to date) example = 'well1033', maxiter = 1.5e4, iter_save = 1e2, 
%             (m,n) = (1033,320), cond(A) = 166
%   (up to date) example = 'ash219', maxiter = 700, iter_save = 10, 
%             (m,n) = (219,85), cond(A) = 3
%   example = 'ash331', maxiter = 700, iter_save = 10, 
%             (m,n) = (331,104), cond(A) = 3
%   (up to date) example = 'ash958', maxiter = 1500, iter_save = 10, 
%             (m,n) = (958,292), cond(A) = 3
%   (up to date) example = 'ash608', maxiter = 1500, iter_save = 10, 
%             (m,n) = (608,188), cond(A) = 3 

% SRK best:
%   (up to date) example = 'illc1033', maxiter = 1.5e4, iter_save = 1e2, 
%             (m,n) = (1033,320), cond(A) = 18888


% SRK as good as SRK-REM and all other methods perform worse:
%   (up to date) example = 'landmark', maxiter = 2e6, iter_save = 1e4, 
%             (m,n) = (71952,2704), cond(A) = 6.8e17
%   (up to date) example = 'Maragal_2', maxiter = 1e5, iter_save = 1e3, 
%             (m,n) = (555,350), cond(A) = 1.5e49
%   example = 'Maragal_3', maxiter = 1e6, iter_save = 1e4, 
%             (m,n) = (1690,860), cond(A) = 2.5e48

writeout = true;

example = 'landmark';
maxiter = 2e6;
iter_save = 1e4;

A = load(['SuiteSparseCollection/data/' example '.mat']).Problem.A;

mkdir(['./SuiteSparseCollection/results/' example]) 

path_for_writeout = ['./SuiteSparseCollection/results/' example '/'];

single_experiment(A, path_for_writeout, maxiter, iter_save, writeout);

fprintf('\n m = %d, n = %d, \n', size(A,1), size(A,2))
fprintf('cond(A) = %f \n', cond(full(A)))