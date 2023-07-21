clear, clc, close all

m = 200;  
n = 500;  
sp = 10;

lambda = 5;

const_betas = [.1:.1:.9, .96, .97];

maxiter = 1e5;
iter_save = 1;
num_repeats = 50;

writeout = true;

res_goals = [1e-2; 1e-4; 1e-6];  % monitor runtime for these residuals, must be decreasing

tol_norm_d = 1e-12;
tol_rel_Bregman_dist = 1e-9; 

rowsamp = 'rownorms_squared';

rand('state', 1)
randn('state', 1)

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
        = deal(zeros(floor(maxiter/iter_save), num_repeats));
[errs_hb, errs_Bregman_hb, res_hb] = deal(zeros(floor(maxiter/iter_save), num_repeats, num_const_betas));

[rt_srk, rt_esrk, rt_hb_inexact, rt_hb_double_inexact, rt_hb_opt_beta] = deal(zeros(maxiter, num_repeats));
rt_hb = zeros(maxiter, num_repeats, num_const_betas);

[rt_for_res_goals_srk, rt_for_res_goals_esrk, rt_for_res_goals_hb_inexact, rt_for_res_goals_hb_double_inexact, rt_for_res_goals_hb_opt_beta]...
    = deal(Inf(length(res_goals), num_repeats));

rt_for_res_goals_hb = Inf(length(res_goals), num_repeats, num_const_betas);

num_res_goals = length(res_goals);



for repeat = 1:num_repeats

    fprintf('repeat #%d\n', repeat)

    [curr_num_res_goal_srk, curr_num_res_goal_esrk, curr_num_res_goal_hb_inexact, curr_num_res_goal_hb_double_inexact, curr_num_res_goal_hb_opt_beta]...
    = deal( 1 );

    curr_num_res_goal_hb = ones(num_const_betas, 1);

    A = randn(m,n);
    x_hat = sparserandn(n, sp);
    b = A * x_hat;
    x_pinv = pinv(A) * b;

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

    % debug
    %S_upper_bound = S_dist(xdual_hb_inexact, x_hat, lambda);
    
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
        rt_srk(iter + 1, repeat) = rt_srk(iter, repeat) + toc;

        % sparse Kaczmarz with exact step size
        tic
        [~,~,t] = linesearch_shrinkage(x_esrk, xdual_esrk, ai, bi, lambda);
        xdual_esrk = xdual_esrk - t * ai;
        x_esrk = S(xdual_esrk);
        rt_esrk(iter + 1, repeat) = rt_esrk(iter, repeat) + toc;

        % sparse Kaczmarz with constant heavy ball momentum
        for ii = 1:num_const_betas
            tic
            beta = const_betas(ii);
            t = (ai' * x_hb(:,ii) - bi) / normA_sqr(i);
            new_xdual_hb_old(:,ii) = xdual_hb(:,ii);
            xdual_hb(:,ii) = xdual_hb(:,ii) - t * ai + beta * (xdual_hb(:,ii) - xdual_hb_old(:,ii));
            xdual_hb_old(:,ii) = new_xdual_hb_old(:,ii);
            x_hb(:,ii) = S(xdual_hb(:,ii));    
            rt_hb(iter + 1, repeat, ii) = rt_hb(iter, repeat, ii) + toc;
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
            rt_hb_inexact(iter + 1, repeat) = rt_hb_inexact(iter, repeat) + toc;
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

            

            rt_hb_double_inexact(iter + 1, repeat) = rt_hb_double_inexact(iter, repeat) + toc;

        % sparse Kaczmarz with minimal error heavy ball momentum
            tic
            t = (ai' * x_hb_opt_beta - bi) / normA_sqr(i);
            ydual = xdual_hb_opt_beta - t * ai;
            d_beta_opt = xdual_hb_opt_beta - xdual_hb_opt_beta_old;
            y = S(ydual); 
            if norm(d_beta_opt) > tol_norm_d
                [~,~,beta_opt] = linesearch_shrinkage(y, ydual, ...
                                            - d_beta_opt, - s_opt_beta, lambda); 
                %test_beta_optimality(ydual, d, s_opt_beta, lambda, beta);
                new_xdual_hb_opt_beta_old = xdual_hb_opt_beta;
                xdual_hb_opt_beta = ydual + beta_opt * d_beta_opt;
                xdual_hb_opt_beta_old = new_xdual_hb_opt_beta_old;
                x_hb_opt_beta = S(xdual_hb_opt_beta);
                s_opt_beta = beta_opt * s_opt_beta - bi * t;
            else
                beta_opt = 0;
                new_xdual_hb_opt_beta_old = xdual_hb_inexact;
                xdual_hb_opt_beta = ydual;
                xdual_hb_opt_beta_old = new_xdual_hb_opt_beta_old;
                x_hb_opt_beta = y;
                s_opt_beta = -bi * t;
            end            
            rt_hb_opt_beta(iter + 1, repeat) = rt_hb_opt_beta(iter, repeat) + toc;
    
        % store errors 
        if mod(iter, iter_save) == 0

            curr_res_srk = norm(A * x_srk - b) / norm_b;
            res_srk(iter_plot, repeat) = curr_res_srk;
            if curr_num_res_goal_srk <= num_res_goals
                if curr_res_srk < res_goals(curr_num_res_goal_srk) 
                   rt_for_res_goals_srk(curr_num_res_goal_srk, repeat) = rt_srk(iter, repeat);
                   curr_num_res_goal_srk = curr_num_res_goal_srk + 1;
                end
            end
            errs_srk(iter, repeat) = norm(x_srk - x_hat) / norm_x_hat;
            errs_Bregman_srk(iter_plot, repeat) = S_dist(xdual_srk, x_hat, lambda) / varphi(x_hat);

            curr_res_esrk = norm(A * x_esrk - b) / norm_b;
            res_esrk(iter_plot, repeat) = curr_res_esrk;
            if curr_num_res_goal_esrk <= num_res_goals
                if curr_res_esrk < res_goals(curr_num_res_goal_esrk) 
                   rt_for_res_goals_esrk(curr_num_res_goal_esrk, repeat) = rt_esrk(iter, repeat);
                   curr_num_res_goal_esrk = curr_num_res_goal_esrk + 1;
                end
            end
            errs_esrk(iter_plot, repeat) = norm(x_esrk - x_hat) / norm_x_hat;
            errs_Bregman_esrk(iter_plot, repeat) = S_dist(xdual_esrk, x_hat, lambda) / varphi(x_hat);

            curr_res_hb_inexact = norm(A * x_hb_inexact - b) / norm_b;
            res_hb_inexact(iter_plot, repeat) = curr_res_hb_inexact;
            if curr_num_res_goal_hb_inexact <= num_res_goals
                if curr_res_hb_inexact < res_goals(curr_num_res_goal_hb_inexact) 
                   rt_for_res_goals_hb_inexact(curr_num_res_goal_hb_inexact, repeat) = rt_hb_inexact(iter, repeat);
                   curr_num_res_goal_hb_inexact = curr_num_res_goal_hb_inexact + 1;
                end
            end
            errs_hb_inexact(iter_plot, repeat) = norm(x_hb_inexact - x_hat) / norm_x_hat;
            errs_Bregman_hb_inexact(iter_plot, repeat) = S_dist(xdual_hb_inexact, x_hat, lambda) / varphi(x_hat);

            curr_res_hb_double_inexact = norm(A * x_hb_double_inexact - b) / norm_b;
            res_hb_double_inexact(iter_plot, repeat) = curr_res_hb_double_inexact; 
            if curr_num_res_goal_hb_double_inexact <= num_res_goals
                if curr_res_hb_double_inexact < res_goals(curr_num_res_goal_hb_double_inexact)
                   rt_for_res_goals_hb_double_inexact(curr_num_res_goal_hb_double_inexact, repeat) = rt_hb_double_inexact(iter, repeat);
                   curr_num_res_goal_hb_double_inexact = curr_num_res_goal_hb_double_inexact + 1;                
                end
            end
            errs_hb_double_inexact(iter_plot, repeat) = norm(x_hb_double_inexact - x_hat) / norm_x_hat;
            errs_Bregman_hb_double_inexact(iter_plot, repeat) = S_dist(xdual_hb_double_inexact, x_hat, lambda) / varphi(x_hat); 

            curr_res_hb_opt_beta = norm(A * x_hb_opt_beta - b) / norm_b;
            res_hb_opt_beta(iter_plot, repeat) = curr_res_hb_opt_beta;
            if curr_num_res_goal_hb_opt_beta <= num_res_goals
                if curr_res_hb_opt_beta < res_goals(curr_num_res_goal_hb_opt_beta) 
                   rt_for_res_goals_hb_opt_beta(curr_num_res_goal_hb_opt_beta, repeat) = rt_hb_opt_beta(iter, repeat);
                   curr_num_res_goal_hb_opt_beta = curr_num_res_goal_hb_opt_beta + 1;                
                end
            end
            errs_hb_opt_beta(iter_plot, repeat) = norm(x_hb_opt_beta - x_hat) / norm_x_hat;
            errs_Bregman_hb_opt_beta(iter_plot, repeat) = S_dist(xdual_hb_opt_beta, x_hat, lambda) / varphi(x_hat);

            for ii = 1:length(const_betas)
                curr_res_hb = norm(A * x_hb(:,ii) - b) / norm_b;
                res_hb(iter_plot, repeat, ii) = curr_res_hb;
                if curr_num_res_goal_hb(ii) <= num_res_goals
                    if curr_res_hb < res_goals(curr_num_res_goal_hb(ii)) 
                        rt_for_res_goals_hb(curr_num_res_goal_hb(ii), repeat, ii) = rt_hb(iter, repeat, ii);
                        curr_num_res_goal_hb(ii) = curr_num_res_goal_hb(ii) + 1;        
                    end 
                end
                errs_hb(iter_plot, repeat, ii) = norm(x_hb(:,ii) - x_hat) / norm_x_hat;
                errs_Bregman_hb(iter_plot, repeat, ii) = S_dist(xdual_hb(:,ii), x_hat, lambda) / varphi(x_hat);
            end
            iter_plot = iter_plot + 1;
        end    

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


%% display runtime at which relative residuals specified in res_goals are reached

[min_rt_for_res_goals_srk, max_rt_for_res_goals_srk, ~, ...
 quant25_rt_for_res_goals_srk, quant75_rt_for_res_goals_srk] = compute_minmax_median_quantiles(rt_for_res_goals_srk); 
mean_rt_for_res_goals_srk = mean(rt_for_res_goals_srk, 2);

[min_rt_for_res_goals_esrk, max_rt_for_res_goals_esrk, ~, ...
 quant25_rt_for_res_goals_esrk, quant75_rt_for_res_goals_esrk] = compute_minmax_median_quantiles(rt_for_res_goals_esrk);
mean_rt_for_res_goals_esrk = mean(rt_for_res_goals_esrk, 2);

[min_rt_for_res_goals_hb_inexact, max_rt_for_res_goals_hb_inexact, ~, ...
 quant25_rt_for_res_goals_hb_inexact, quant75_rt_for_res_goals_hb_inexact] = compute_minmax_median_quantiles(rt_for_res_goals_hb_inexact);
mean_rt_for_res_goals_hb_inexact = mean(rt_for_res_goals_hb_inexact, 2); 

[min_rt_for_res_goals_hb_double_inexact, max_rt_for_res_goals_hb_double_inexact, ~, ...
 quant25_rt_for_res_goals_hb_double_inexact, quant75_rt_for_res_goals_hb_double_inexact] = compute_minmax_median_quantiles(rt_for_res_goals_hb_double_inexact);
mean_rt_for_res_goals_hb_double_inexact = mean(rt_for_res_goals_hb_double_inexact, 2);

[min_rt_for_res_goals_hb_opt_beta, max_rt_for_res_goals_hb_opt_beta, ~, ...
 quant25_rt_for_res_goals_hb_opt_beta, quant75_rt_for_res_goals_hb_opt_beta] = compute_minmax_median_quantiles(rt_for_res_goals_hb_opt_beta);
mean_rt_for_res_goals_hb_opt_beta = mean(rt_for_res_goals_hb_opt_beta, 2);

% beta = .5
[min_rt_for_res_goals_hb_medium_beta, max_rt_for_res_goals_hb_medium_beta, ~, ...
 quant25_rt_for_res_goals_hb_medium_beta, quant75_rt_for_res_goals_hb_medium_beta] = compute_minmax_median_quantiles(rt_for_res_goals_hb(:,5));
mean_rt_for_res_goals_hb_medium_beta = mean(rt_for_res_goals_hb(:,:,5), 2);

% beta = .96
[min_rt_for_res_goals_hb_large_beta, max_rt_for_res_goals_hb_large_beta, ~, ...
 quant25_rt_for_res_goals_hb_large_beta, quant75_rt_for_res_goals_hb_large_beta] = compute_minmax_median_quantiles(rt_for_res_goals_hb(:,end-1));
mean_rt_for_res_goals_hb_large_beta = mean(rt_for_res_goals_hb(:,end-1), 2);

% beta = .97
[min_rt_for_res_goals_hb_too_large_beta, max_rt_for_res_goals_hb_too_large_beta, median_rt_for_res_goals_hb_too_large_beta, ...
 quant25_rt_for_res_goals_hb_too_large_beta, quant75_rt_for_res_goals_hb_too_large_beta] = compute_minmax_median_quantiles(rt_for_res_goals_hb(:,end));
mean_rt_for_res_goals_hb_too_large_beta = mean(rt_for_res_goals_hb(:,end), 2); 

[min_rt_for_res_goals_hb, max_rt_for_res_goals_hb, mean_rt_for_res_goals_hb, ...
    quant25_rt_for_res_goals_hb, quant75_rt_for_res_goals_hb] ...
    = deal(zeros(num_res_goals, num_const_betas));
for ii = 1:num_const_betas
    [min_rt_for_res_goals_hb(:,ii), max_rt_for_res_goals_hb(:,ii), ~, ...
     quant25_rt_for_res_goals_hb(:,ii), quant75_rt_for_res_goals_hb(:,ii)] = compute_minmax_median_quantiles(rt_for_res_goals_hb(:,:,ii));
    mean_rt_for_res_goals_hb(:,ii) = mean(rt_for_res_goals_hb(:,:,ii), 2); 
end

% sparse Kaczmarz
fprintf('Runtime for SRK (min/mean/max), corresponding log-residuals in first column \n')
disp([log10(res_goals), min_rt_for_res_goals_srk, mean_rt_for_res_goals_srk, max_rt_for_res_goals_srk])

% exact step sparse Kaczmarz
disp('Runtime for ESRK (min/mean/max), corresponding log-residuals in first column \n')
disp([log10(res_goals), min_rt_for_res_goals_esrk, mean_rt_for_res_goals_esrk, max_rt_for_res_goals_esrk])

% sparse Kaczmarz with relaxed exact beta 
%disp('Runtime for SRKREM (min/mean/max), corresponding log-residuals in first column \n')
%disp([log10(res_goals), min_rt_for_res_goals_hb_inexact, mean_rt_for_res_goals_hb_inexact, max_rt_for_res_goals_hb_inexact])

% sparse Kaczmarz with relaxed exact (t, beta) 
disp('Runtime for SRKREM (min/mean/max), corresponding log-residuals in first column \n')
disp([log10(res_goals), min_rt_for_res_goals_hb_double_inexact, mean_rt_for_res_goals_hb_double_inexact, max_rt_for_res_goals_hb_double_inexact])

% sparse Kaczmarz with exact momentum 
disp('Runtime for SRKEM (min/mean/max), corresponding log-residuals in first column \n')
disp([log10(res_goals), min_rt_for_res_goals_hb_opt_beta, mean_rt_for_res_goals_hb_opt_beta, max_rt_for_res_goals_hb_opt_beta])

% sparse Kaczmarz with beta = 0.5
disp('Runtime for HB-SRK with beta = 0.5 (min/mean/max), corresponding log-residuals in first column \n')
disp([log10(res_goals), min_rt_for_res_goals_hb_medium_beta, mean_rt_for_res_goals_hb_medium_beta, max_rt_for_res_goals_hb_medium_beta])

% sparse Kaczmarz with beta = 0.96
disp('Runtime for HB-SRK with beta = 0.96 (min/mean/max), corresponding log-residuals in first column \n')
disp([log10(res_goals), min_rt_for_res_goals_hb_large_beta, mean_rt_for_res_goals_hb_large_beta, max_rt_for_res_goals_hb_large_beta])

% sparse Kaczmarz with beta = 0.97
disp('Runtime for HB-SRK with beta = 0.97 (min/mean/max), corresponding log-residuals in first column \n')
disp([log10(res_goals), min_rt_for_res_goals_hb_too_large_beta, mean_rt_for_res_goals_hb_too_large_beta, max_rt_for_res_goals_hb_too_large_beta])




if writeout

    writeout_data_over_array_on_xaxis(...
        './randn_sparse/lambda=5/rt_for_res_goals.txt',...
        {'res_goals', 'min_rt_for_res_goals_srk', 'min_rt_for_res_goals_esrk', 'min_rt_for_res_goals_hb_double_inexact', 'min_rt_for_res_goals_hb_opt_beta', 'min_rt_for_res_goals_hb_large_beta', 'min_rt_for_res_goals_hb_too_large_beta', ...
                      'max_rt_for_res_goals_srk', 'max_rt_for_res_goals_esrk', 'max_rt_for_res_goals_hb_double_inexact', 'max_rt_for_res_goals_hb_opt_beta', 'max_rt_for_res_goals_hb_large_beta', 'max_rt_for_res_goals_hb_too_large_beta', ...
                      'median_rt_for_res_goals_srk', 'median_rt_for_res_goals_esrk', 'median_rt_for_res_goals_hb_inexact', 'median_rt_for_res_goals_hb_double_inexact', ...
                      'median_rt_for_res_goals_hb_opt_beta', 'median_rt_for_res_goals_hb_large_beta', 'median_rt_for_res_goals_hb_too_large_beta'}, ...
         res_goals, ...
         [min_rt_for_res_goals_srk, min_rt_for_res_goals_esrk, min_rt_for_res_goals_hb_double_inexact, min_rt_for_res_goals_hb_opt_beta, min_rt_for_res_goals_hb_large_beta, min_rt_for_res_goals_hb_too_large_beta, ...
          max_rt_for_res_goals_srk, max_rt_for_res_goals_esrk, max_rt_for_res_goals_hb_double_inexact, max_rt_for_res_goals_hb_opt_beta, max_rt_for_res_goals_hb_large_beta, max_rt_for_res_goals_hb_too_large_beta, ...
          mean_rt_for_res_goals_srk, mean_rt_for_res_goals_esrk, mean_rt_for_res_goals_hb_double_inexact, mean_rt_for_res_goals_hb_opt_beta, mean_rt_for_res_goals_hb_large_beta, mean_rt_for_res_goals_hb_too_large_beta]) 

end


