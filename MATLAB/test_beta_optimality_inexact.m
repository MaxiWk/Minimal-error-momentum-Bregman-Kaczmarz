function test_beta_optimality_inexact(y, d, s, beta_opt)

    figure

    g = @(beta) 0.5 * beta^2 * (d' * d) + beta * y' * d - s*beta;

    beta_grid = linspace(beta_opt - 1, beta_opt + 1, 100);
    g_grid = zeros(size(beta_grid));
    g_opt = g(beta_opt);

    for ii = 1:length(beta_grid)
        g_grid(ii) = g(beta_grid(ii));  
    end

    hold on
    plot(beta_grid, g_grid)
    scatter(beta_opt, g_opt, '*')
    title('beta-optimality for fixed t')
    hold off

end