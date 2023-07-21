function test_double_inexact_optimality(x, x_hat, ai, d, t_opt, beta_opt)

    g = @(t, beta) norm(x - t * ai + beta * d - x_hat)^2;

    g_opt = g(t_opt, beta_opt);

    % test t - optimality 
    figure
    t_grid = linspace(t_opt - 1, t_opt + 1, 100);
    g_grid = zeros(size(t_grid));
    for ii = 1:length(t_grid)
        g_grid(ii) = g(t_grid(ii), beta_opt);  
    end
    hold on
    plot(t_grid, g_grid)
    scatter(t_opt, g_opt, '*')
    title('t-optimality for fixed beta')
    hold off

    % test beta - optimality 
    figure
    beta_grid = linspace(beta_opt - 1, beta_opt + 1, 100);
    for ii = 1:length(beta_grid)
        g_grid(ii) = g(t_opt, beta_grid(ii));  
    end
    hold on
    plot(beta_grid, g_grid)
    scatter(beta_opt, g_opt, '*')
    title('beta-optimality for fixed t')
    hold off

end