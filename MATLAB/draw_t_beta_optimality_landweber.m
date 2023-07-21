function draw_t_beta_optimality_landweber(x, d, x_hat, r2, t, beta)

    update_opt = x - t * r2 + beta * d;

    %% draw t

    t_grid = linspace(t-0.1, t+0.1, 100);
    
    dist_sqr_for_t = zeros(size(t_grid));

    for i = 1:length(t_grid)
        update = x - t_grid(i) * r2 + beta * d;
        dist_sqr_for_t(i) = norm(update - x_hat)^2; 
    end 

    figure
    plot(t_grid, dist_sqr_for_t)
    hold on
    dist_sqr_opt = norm(update_opt-x_hat)^2;
    scatter(t, dist_sqr_opt, '*')
    title('t-optimality for fixed optimal beta')
    hold off


    %% draw beta
    
    if abs(beta) < 1e-5
        beta_grid = linspace(-1, 1, 100);
    else
        beta_grid = linspace(beta-2*abs(beta), beta+2*abs(beta), 100);
    end
    
    dist_sqr_for_beta = zeros(size(beta_grid));

    for i = 1:length(beta_grid)
        update = x - t * r2 + beta_grid(i) * d; 
        dist_sqr_for_beta(i) = norm(update - x_hat)^2; 
    end 

    figure
    plot(beta_grid, dist_sqr_for_beta)
    hold on


    dist_sqr_opt = norm(update_opt-x_hat)^2;
    scatter(beta, dist_sqr_opt, '*')
    title('beta-optimality for fixed optimal t')
    hold off

end