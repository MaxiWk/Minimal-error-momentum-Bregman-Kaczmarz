function [x, last_residual] = linearized_Bregman(A, b, lambda, maxiter)

    % define soft shrinkage function
    S = @(x) sign(x) .* max(0, abs(x)-lambda);

    [xdual, x] = deal( zeros(size(A,2)) );

    t = 1/norm(A)^2;

    for iter = 1:maxiter
        xdual = xdual - t * A'*(A*x-b);
        x = S(xdual);
    end

    last_residual = norm(A * x - b);

end