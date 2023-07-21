addpath AIRtools_1.0

writeout = true;

N = 30;
theta = 0:2:359;          % Angles
%p = round(sqrt(2)*N)*0.5; % Number of rays per angle (round(sqrt(2)*N) would be "full sampling"
p = round(sqrt(2)*N)*0.5; % Number of rays per angle (round(sqrt(2)*N) would be "full sampling"
d = sqrt(2)*N;
%R = sqrt(2)*N;            % Distance between extremal rays (parallel beam)

num_sweeps = 100; % Number of "Sweeps"

A = paralleltomo(N,theta,p,d);

maxiter = size(A,1) * num_sweeps;

iter_save = floor(maxiter / 100);

path_for_writeout = './CT/results/paralleltomo/';

single_experiment(A, path_for_writeout, maxiter, iter_save, writeout);

fprintf('\n m = %d, n = %d, \n', size(A,1), size(A,2))
fprintf('cond(A) = %f \n', cond(full(A)))