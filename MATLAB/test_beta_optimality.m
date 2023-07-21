function test_beta_optimality(ydual, d, s, lambda, beta_opt)

    figure

    S = @(x) sign(x) .* max(0, abs(x)-lambda);
    g = @(beta) 0.5*norm(S(ydual + beta * d))^2 - s*beta;

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