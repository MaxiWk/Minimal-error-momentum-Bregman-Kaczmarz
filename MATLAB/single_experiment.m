function single_experiment(A, path_for_writeout, maxiter, iter_save, writeout)

    rand('state', 1)
    randn('state', 1)
    
    n = size(A,2);
    sp = floor(0.1 * n);
    x_hat = sparserandn(n, sp);
    b = A * x_hat;
    
    lambda = 1;   
    
    const_betas = .1:.1:.6; % divergence for beta = .7
    
    tol_norm_d = 1e-6;
    tol_rel_Bregman_dist = 1e-9;
    
    rowsamp = 'rownorms_squared';
    
    % define soft shrinkage function
    S = @(x) sign(x) .* max(0, abs(x)-lambda);
    varphi = @(x) lambda * norm(x,1) + 0.5 * (x'*x);
    
    num_const_betas = length(const_betas);
    
    % initialize storage array for errors
    [errs_srk, errs_Bregman_srk, res_srk,...
     errs_esrk, errs_Bregman_esrk, res_esrk,...   
     errs_hb_inexact, errs_Bregman_hb_inexact, res_hb_inexact, ...
     errs_hb_double_inexact, errs_Bregman_hb_double_inexact, res_hb_double_inexact, ...
     errs_hb_opt_beta, errs_Bregman_hb_opt_beta, res_hb_opt_beta]... 
            = deal(zeros(floor(maxiter/iter_save), 1));
    [errs_hb, errs_Bregman_hb, res_hb] = deal(zeros(floor(maxiter/iter_save), num_const_betas));
    
    [rt_srk, rt_esrk, rt_hb_inexact, rt_hb_double_inexact, rt_hb_opt_beta] = deal(zeros(maxiter, 1));
    rt_hb = zeros(maxiter, 1, num_const_betas);
    
    
    
    
    % precomputations
    normA_sqr = vecnorm(A,2,2).^2; % precompute squared norms of A
    norm_b = norm(b);
    norm_x_hat = norm(x_hat);
    
    %maxiter_lin_Breg = 1e3;
    %[x_hat, last_residual] = linearized_Bregman(A, b, lambda, maxiter_lin_Breg);
    
    % initialization
    xdual_srk = zeros(n,1); 
    x_srk = S(xdual_srk); 
    xdual_esrk = xdual_srk;
    x_esrk = x_srk;
    xdual_hb_inexact = xdual_srk;
    x_hb_inexact = S(xdual_hb_inexact); 
    xdual_hb_double_inexact = xdual_srk;
    x_hb_double_inexact = S(xdual_hb_double_inexact);
    xdual_hb_opt_beta = xdual_srk;
    x_hb_opt_beta = x_srk;
    [xdual_hb, x_hb] = deal( zeros(n, num_const_betas) );
    for ii = 1:num_const_betas
        xdual_hb(:,ii) = xdual_srk;
        x_hb(:,ii) = x_srk;
    end
    xdual_hb_old = xdual_hb;
    new_xdual_hb_old = xdual_hb_old;
    xdual_hb_inexact_old = xdual_hb_inexact;
    xdual_hb_double_inexact_old = xdual_hb_double_inexact;
    xdual_hb_opt_beta_old = xdual_hb_opt_beta;
    s_opt_beta = 0;
    s_inexact = 0;
    s_double_inexact = 0;
    iter_plot = 1;
    
    % define function for row sampling
    switch lower(rowsamp)
        case {'rownorms_squared'}
            p = normA_sqr./sum(normA_sqr);
            P = cumsum(p);
            sampling = @() nnz(rand>P)+1;
        case {'uniform'}
            sampling = @() randi(m,1);
        case {'random_probabilies'}
            p = rand(m,1); p=p/sum(p);
            P = cumsum(p);
            sampling = @() nnz(rand>P)+1;
    end 
       
    for iter = 1:maxiter

        if mod(10*iter, maxiter) == 0
            fprintf('%d/%d iterations \n', iter, maxiter)
        end
    
        %if mod(iter, floor(maxiter/10)) == 0
        %    fprintf('Iteration %d/%d\n', iter, maxiter)
        %end
        %if iter == maxiter
        %    fprintf('\n')
        %end
    
        i = sampling();
        ai = A(i,:)';
        bi = b(i);
        
        % sparse Kaczmarz
        tic
        t = (ai' * x_srk - bi) / normA_sqr(i);
        xdual_srk = xdual_srk - t * ai;
        x_srk = S(xdual_srk); 
        rt_srk(iter + 1) = rt_srk(iter) + toc;
    
        % sparse Kaczmarz with exact step size
            % debug 
            %xdual_esrk_old = xdual_esrk;
            %old_dist = S_dist(xdual_esrk, x_hat, lambda);
        tic
        [x_esrk, xdual_esrk, ~] = linesearch_shrinkage(x_esrk, xdual_esrk, ai, bi, lambda);
        rt_esrk(iter + 1) = rt_esrk(iter) + toc;
    
            % debug 
            %test_t_optimality(xdual_esrk_old, ai, bi, lambda, t)
            %true_dist = S_dist(xdual_esrk, x_hat, lambda);
            %upper_bound = old_dist - 0.5 * (ai'*x_esrk-bi)/norm(ai)^2;
            %must_be_nonnegative = upper_bound - true_dist;
            %if must_be_nonnegative < -1e-6
            %    fprintf('Bug in esrk, must_be_nonnegative = %f\n', must_be_nonnegative)
            %end
    
        % sparse Kaczmarz with constant heavy ball momentum
        for ii = 1:num_const_betas
            tic
            beta = const_betas(ii);
            t = (ai' * x_hb(:,ii) - bi) / normA_sqr(i);
            new_xdual_hb_old(:,ii) = xdual_hb(:,ii);
            xdual_hb(:,ii) = xdual_hb(:,ii) - t * ai + beta * (xdual_hb(:,ii) - xdual_hb_old(:,ii));
            xdual_hb_old(:,ii) = new_xdual_hb_old(:,ii);
            x_hb(:,ii) = S(xdual_hb(:,ii));    
            rt_hb(iter + 1, ii) = rt_hb(iter, ii) + toc;
        end
    
        % sparse Kaczmarz with inexact heavy ball momentum 
        %if S_dist(xdual_hb_inexact, x_hat, lambda) / varphi(x_hat) > tol_rel_Bregman_dist
            tic
            t = (ai' * x_hb_inexact - bi) / normA_sqr(i);
            d_inexact = xdual_hb_inexact - xdual_hb_inexact_old;
            norm_d = norm(d_inexact);
            ydual = xdual_hb_inexact - t * ai;
            y = S(ydual);
            %disp( abs(s_inexact - (xdual_hb_inexact - xdual_hb_inexact_old)'*x_pinv) )        
            if norm_d > tol_norm_d
                beta_inexact = (s_inexact - d_inexact' * y)/norm_d^2;
                %test_beta_optimality_inexact(y, d_inexact, s_inexact, beta_inexact);
                new_xdual_hb_inexact_old = xdual_hb_inexact;
                xdual_hb_inexact = ydual + beta_inexact * d_inexact;
                xdual_hb_inexact_old = new_xdual_hb_inexact_old;
                x_hb_inexact = S(xdual_hb_inexact);
                s_inexact = beta_inexact * s_inexact - bi * t;
            else
                beta_inexact = 0;
                new_xdual_hb_inexact_old = xdual_hb_inexact;
                xdual_hb_inexact = ydual;
                x_hb_inexact = y;
                s_inexact = -bi * t;
            end
            rt_hb_inexact(iter + 1) = rt_hb_inexact(iter) + toc;
        %end
    
        % sparse Kaczmarz with relaxed exact momentum ('double inexact')
            tic
            d_double_inexact = xdual_hb_double_inexact - xdual_hb_double_inexact_old;
            norm_d_sqr = d_double_inexact' * d_double_inexact;
            ai_d = ai' * d_double_inexact;
            dist_dep = normA_sqr(i) * norm_d_sqr - ai_d^2;
            r = ai' * x_hb_double_inexact - bi;
            s_diff_x_d = s_double_inexact - x_hb_double_inexact' * d_double_inexact;
            if dist_dep > tol_norm_d^2
                t = (norm_d_sqr * r + ai_d * s_diff_x_d) / dist_dep;
                beta = (ai_d * r + normA_sqr(i) * s_diff_x_d) / dist_dep;
                %test_double_inexact_optimality(x_hb_double_inexact, x_hat, ai, d_double_inexact, t, beta) % debug                 
                new_xdual_hb_double_inexact_old = xdual_hb_double_inexact;
                xdual_hb_double_inexact = xdual_hb_double_inexact - t * ai + beta * d_double_inexact;
                xdual_hb_double_inexact_old = new_xdual_hb_double_inexact_old;
                x_hb_double_inexact = S(xdual_hb_double_inexact); 
                s_double_inexact = beta * s_double_inexact - bi * t;
            else % perform vanilla sparse Kaczmarz
                t = r/normA_sqr(i);
                % beta = 0;
                new_xdual_hb_double_inexact_old = xdual_hb_double_inexact;
                xdual_hb_double_inexact = xdual_hb_double_inexact - t * ai;
                xdual_hb_double_inexact_old = new_xdual_hb_double_inexact_old;
                x_hb_double_inexact = S(xdual_hb_double_inexact); 
                s_double_inexact = - bi * t;
            end
    
            
    
            rt_hb_double_inexact(iter + 1) = rt_hb_double_inexact(iter) + toc;
    
        % sparse Kaczmarz with exact heavy ball momentum and SRK step size
            tic 
            t = (ai' * x_hb_opt_beta - bi) / normA_sqr(i);
            ydual = xdual_hb_opt_beta - t * ai;
            d_beta_opt = xdual_hb_opt_beta - xdual_hb_opt_beta_old;
            y = S(ydual); 
              % debug
              %  old_dist = S_dist(xdual_hb_opt_beta, x_hat, lambda);
            if norm(d_beta_opt) > tol_norm_d
                new_xdual_hb_opt_beta_old = xdual_hb_opt_beta; 
                [x_hb_opt_beta, xdual_hb_opt_beta, beta_opt] = linesearch_shrinkage(y, ydual, ...
                                            - d_beta_opt, - s_opt_beta, lambda); 
                %test_beta_optimality(ydual, d_beta_opt, s_opt_beta, lambda, beta_opt);
                xdual_hb_opt_beta_old = new_xdual_hb_opt_beta_old;
                s_opt_beta = beta_opt * s_opt_beta - bi * t;
            else
                %beta_opt = 0; 
                new_xdual_hb_opt_beta_old = xdual_hb_opt_beta; 
                xdual_hb_opt_beta = ydual;
                xdual_hb_opt_beta_old = new_xdual_hb_opt_beta_old;
                x_hb_opt_beta = y;
                s_opt_beta = -bi * t; 
            end            
            rt_hb_opt_beta(iter + 1) = rt_hb_opt_beta(iter) + toc;

                % debug 
                
                %true_dist = S_dist(xdual_hb_opt_beta, x_hat, lambda);
                %upper_bound = old_dist - 0.5 * (ai'*x_hb_opt_beta-bi)^2/norm(ai)^2;
                %must_be_nonnegative = (upper_bound - true_dist) / true_dist;
                %if must_be_nonnegative < -1e-3
                %    fprintf('Bug in hb_opt_beta, must_be_nonnegative = %f\n', must_be_nonnegative)
                %end
                
    
        % store errors 
        if mod(iter, iter_save) == 0
            res_srk(iter_plot) = norm(A * x_srk - b) / norm_b;
            errs_srk(iter_plot) = norm(x_srk - x_hat) / norm_x_hat;
            errs_Bregman_srk(iter_plot) = S_dist(xdual_srk, x_hat, lambda) / varphi(x_hat);
            res_esrk(iter_plot) = norm(A * x_esrk - b) / norm_b;
            errs_esrk(iter_plot) = norm(x_esrk - x_hat) / norm_x_hat;
            errs_Bregman_esrk(iter_plot) = S_dist(xdual_esrk, x_hat, lambda) / varphi(x_hat);
            res_hb_inexact(iter_plot) = norm(A * x_hb_inexact - b) / norm_b;
            errs_hb_inexact(iter_plot) = norm(x_hb_inexact - x_hat) / norm_x_hat;
            errs_Bregman_hb_inexact(iter_plot) = S_dist(xdual_hb_inexact, x_hat, lambda) / varphi(x_hat);
            res_hb_double_inexact(iter_plot) = norm(A * x_hb_double_inexact - b) / norm_b;
            errs_hb_double_inexact(iter_plot) = norm(x_hb_double_inexact - x_hat) / norm_x_hat;
            errs_Bregman_hb_double_inexact(iter_plot) = S_dist(xdual_hb_double_inexact, x_hat, lambda) / varphi(x_hat);            
            res_hb_opt_beta(iter_plot) = norm(A * x_hb_opt_beta - b) / norm_b;
            errs_hb_opt_beta(iter_plot) = norm(x_hb_opt_beta - x_hat) / norm_x_hat;
            errs_Bregman_hb_opt_beta(iter_plot) = S_dist(xdual_hb_opt_beta, x_hat, lambda) / varphi(x_hat);
            for ii = 1:length(const_betas)
                res_hb(iter_plot, ii) = norm(A * x_hb(:,ii) - b) / norm_b;
                errs_hb(iter_plot, ii) = norm(x_hb(:,ii) - x_hat) / norm_x_hat;
                errs_Bregman_hb(iter_plot, ii) = S_dist(xdual_hb(:,ii), x_hat, lambda) / varphi(x_hat);
            end
            iter_plot = iter_plot + 1;
        end    
    
    end 
    
    
    
    % preprocessing for plot
    
    num_iter_array = (1:iter_save:maxiter)';
    rt_srk = rt_srk(num_iter_array,:);
    rt_esrk = rt_esrk(num_iter_array,:);
    rt_hb_inexact = rt_hb_inexact(num_iter_array,:);
    rt_hb_double_inexact = rt_hb_double_inexact(num_iter_array,:);
    rt_hb_opt_beta = rt_hb_opt_beta(num_iter_array,:);
    rt_hb = rt_hb(num_iter_array,:,:);
    
    
    %% plot errors
    
    % plot residuals over iterations
    
    subplot(1,2,1)
    
    % sparse Kaczmarz
    srk = semilogy(num_iter_array, res_srk, 'Color', 'black');
    
    hold on
    
    % sparse Kaczmarz with exact step size
    esrk = semilogy(num_iter_array, res_esrk, 'Color', 'green');
    
    % sparse Kaczmarz with inexact heavy ball momentum
    %inexact_beta = semilogy(num_iter_array, mean_res_hb_inexact, 'Color', 'magenta');
    
    % sparse Kaczmarz with both inexact step size and heavy ball momentum
    double_inexact = semilogy(num_iter_array, res_hb_double_inexact, 'Color', 'red');
    
    % sparse Kaczmarz with minimum error heavy ball momentum 
    min_err_beta = semilogy(num_iter_array, res_hb_opt_beta, 'Color', 'blue');
    
    % sparse Kaczmarz with constant heavy ball momentum
    
    %{
    for ii = 1:num_const_betas
        color = const_betas(ii) * [0 1 1]; % interp. between black (beta=0) and cyan (beta=1)
        if ii == 1
            const_momentum = semilogy(num_iter_array, res_hb(:, ii), 'Color', color);
        else
            semilogy(num_iter_array, res_hb(:, ii), 'Color', color);
        end
    end
    %}
    
    %{
    legend([srk, esrk, double_inexact, min_err_beta, const_momentum], ...
            'SRK', 'ESRK', 'relaxed exact momentum', ...
            'exact momentum with SRK step size (SRKEM)', ...
            'momentum with constant parameters');
    %}
    
    legend([srk, esrk, double_inexact, min_err_beta], ...
            'SRK', 'ESRK', 'relaxed exact momentum', ...
            'exact momentum with SRK step size (SRKEM)');
    
            
    
    title('Rel. res. over iterations')
    
    hold off
    
    
    
    
    % plot residuals over runtime
    
    subplot(1,2,2)
    
    % sparse Kaczmarz
    srk = semilogy(rt_srk, res_srk, 'Color', 'black');
    
    hold on
    
    % sparse Kaczmarz with exact step size
    esrk = semilogy(rt_esrk, res_esrk, 'Color', 'green');
    
    % sparse Kaczmarz with inexact heavy ball momentum
    %inexact_beta = semilogy(num_iter_array, mean_res_hb_inexact, 'Color', 'magenta');
    
    % sparse Kaczmarz with both inexact step size and heavy ball momentum
    double_inexact = semilogy(rt_hb_double_inexact, res_hb_double_inexact, 'Color', 'red');
    
    % sparse Kaczmarz with minimum error heavy ball momentum 
    min_err_beta = semilogy(rt_hb_opt_beta, res_hb_opt_beta, 'Color', 'blue');
    
    % sparse Kaczmarz with constant heavy ball momentum
    %{
    for ii = 1:num_const_betas
        color = const_betas(ii) * [0 1 1]; % interp. between black (beta=0) and cyan (beta=1)
        if ii == 1
            const_momentum = semilogy(rt_hb(:,ii), res_hb(:, ii), 'Color', color);
        else
            semilogy(rt_hb(:,ii), res_hb(:, ii), 'Color', color);
        end
    end
    %}
    
    %{
    legend([srk, esrk, double_inexact, min_err_beta, const_momentum], ...
            'SRK', 'ESRK', 'relaxed exact momentum', ...
            'exact momentum with SRK step size (SRKEM)', ...
            'momentum with constant parameters');
    %}
    
    legend([srk, esrk, double_inexact, min_err_beta], ...
            'SRK', 'ESRK', 'relaxed exact momentum', ...
            'exact momentum with SRK step size (SRKEM)');
    
    title('Rel. res. over runtime')
    
    hold off



    
    % plot Bregman distance to exact solution over iterations
    %{
    figure
    
    % sparse Kaczmarz
    srk = semilogy(num_iter_array, errs_Bregman_srk, 'Color', 'black');
    
    hold on
    
    % sparse Kaczmarz with exact step size
    esrk = semilogy(num_iter_array, errs_Bregman_esrk, 'Color', 'green');
    
    % sparse Kaczmarz with inexact heavy ball momentum
    %inexact_beta = semilogy(num_iter_array, mean_res_hb_inexact, 'Color', 'magenta');
    
    % sparse Kaczmarz with both inexact step size and heavy ball momentum
    double_inexact = semilogy(num_iter_array, errs_Bregman_hb_double_inexact, 'Color', 'red');
    
    % sparse Kaczmarz with minimum error heavy ball momentum 
    min_err_beta = semilogy(num_iter_array, errs_Bregman_hb_opt_beta, 'Color', 'blue');
    
    % sparse Kaczmarz with constant heavy ball momentum
    
    for ii = 1:num_const_betas
        color = const_betas(ii) * [0 1 1]; % interp. between black (beta=0) and cyan (beta=1)
        if ii == 1
            const_momentum = semilogy(num_iter_array, errs_Bregman_hb(:, ii), 'Color', color);
        else
            semilogy(num_iter_array, res_hb(:, ii), 'Color', color);
        end
    end
    
    legend([srk, esrk, double_inexact, min_err_beta, const_momentum], ...
            'SRK', 'ESRK', 'relaxed exact momentum', ...
            'exact momentum with SRK step size (SRKEM)', 'momentum with constant parameters');
            
    
    title('Bregman distance to sparse solution over iterations')
    
    hold off
    %}
    
    
    
    if writeout
    
        writeout_data_over_array_on_xaxis(...
        [path_for_writeout 'res_over_iter.txt'],...
        {'k', ...
        'rt_srk', 'rt_esrk', 'rt_hb_double_inexact', 'rt_hb_opt_beta' ...
              'rt_hb_beta_0.1', 'rt_hb_beta_0.2', 'rt_hb_beta_0.3', ...
              'rt_hb_beta_0.4', 'rt_hb_beta_0.5','rt_hb_beta_0.6', ...
        'res_srk', 'res_esrk', 'res_hb_double_inexact', 'res_hb_opt_beta' ...
              'res_hb_beta_0.1', 'res_hb_beta_0.2', 'res_hb_beta_0.3', ...
              'res_hb_beta_0.4', 'res_hb_beta_0.5','res_hb_beta_0.6'}, ...
        num_iter_array, ...
        [rt_srk, rt_esrk, rt_hb_double_inexact, rt_hb_opt_beta, ...
         rt_hb(:, 1), rt_hb(:, 2), rt_hb(:, 3), rt_hb(:, 4), rt_hb(:, 5), ...
         rt_hb(:, 6), ...
         res_srk, res_esrk, res_hb_double_inexact, res_hb_opt_beta, ...
         res_hb(:, 1), res_hb(:, 2), res_hb(:, 3), res_hb(:, 4), res_hb(:, 5), ...
         res_hb(:, 6)]) 
    
    end

end

