function dist = S_dist(xdual, y, lambda)
    s = sign(xdual) .* max(0, abs(xdual) - lambda);
    dist = 0.5 * (s'*s) - xdual' * y + lambda * norm(y, 1) + 0.5 * (y'*y);
end 