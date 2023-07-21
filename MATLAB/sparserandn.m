function x = sparserandn(n,s)
%function x = sparserandn(n,s)
%  Produces an column vector of length n with s non-zero entries at random
%  positions and with normally distributed entries (via randn).

% Dirk Lorenz, d.lorenz@tu-braunschweig.de, 18.05.2017

x = zeros(n,1);
p = randperm(n);

x(p(1:s)) = randn(s,1);
