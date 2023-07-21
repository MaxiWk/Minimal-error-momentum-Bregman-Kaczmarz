function draw_beta_optimality(x, x_old, x_hat, ai, bi, beta)

    figure
    axs_beta_optimality = axes;

    if abs(beta) < 1e-5
        beta_grid = linspace(-1, 1, 100);
    else
        beta_grid = linspace(beta-2*abs(beta), beta+2*abs(beta), 100);
    end
    
    dist_sqr = zeros(size(beta_grid));

    p = x - (ai'*x-bi)/norm(ai)^2 * ai;

    for i = 1:length(beta_grid)
        update = p + beta_grid(i) * (x-x_old);
        dist_sqr(i) = norm(update-x_hat)^2;
    end

    plot(axs_beta_optimality, beta_grid, dist_sqr)
    hold on

    update_opt = p + beta * (x-x_old);
    dist_sqr_opt = norm(update_opt-x_hat)^2;
    scatter(axs_beta_optimality, beta, dist_sqr_opt, '*')

    title('beta-optimality')
    hold off

end