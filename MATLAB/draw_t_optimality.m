function draw_t_optimality(x, x_hat, r2, t)

    update_opt = x - t*r2;

    %% draw t

    if abs(t) < 1
        t_grid = linspace(t-0.01, t+0.01, 100);
    else
        t_grid = linspace(t-1, t+1, 100);
    end
    
    dist_sqr_for_t = zeros(size(t_grid));

    for i = 1:length(t_grid)
        update = x - t_grid(i) * r2;
        dist_sqr_for_t(i) = norm(update - x_hat)^2; 
    end 

    figure 
    plot(t_grid, dist_sqr_for_t)
    hold on
    dist_sqr_opt = norm(update_opt-x_hat)^2;
    scatter(t, dist_sqr_opt, '*')
    title('t-optimality for fixed optimal beta')
    hold off



end