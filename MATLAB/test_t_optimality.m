function test_t_optimality(z, ai, bi, lambda, t_opt)

    figure

    S = @(x) sign(x) .* max(0, abs(x)-lambda);
    g = @(t) 0.5*norm(S(z - t * ai))^2 + t * bi;

    t_grid = linspace(t_opt - 1, t_opt + 1, 100); 
    g_grid = zeros(size(t_grid));
    g_opt = g(t_opt);

    for ii = 1:length(t_grid) 
        g_grid(ii) = g(t_grid(ii));  
    end

    hold on
    plot(t_grid, g_grid)
    scatter(t_opt, g_opt, '*')
    title('t-optimality with linesearch-shrinkage')
    hold off

end