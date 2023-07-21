clear, clc, close all

m = 200;  
n = 500;  
sp = 10;  

lambda = 0.1;  

const_betas = .1:.1:.6;

maxiter = 1e5;
iter_save = 1e3;  
num_repeats = 50;  

writeout = true; 

tol_norm_d = eps; 
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



for repeat = 1:num_repeats

    fprintf('repeat #%d\n', repeat)

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
            res_srk(iter_plot, repeat) = norm(A * x_srk - b) / norm_b;
            errs_srk(iter_plot, repeat) = norm(x_srk - x_hat) / norm_x_hat;
            errs_Bregman_srk(iter_plot, repeat) = S_dist(xdual_srk, x_hat, lambda) / varphi(x_hat);
            res_esrk(iter_plot, repeat) = norm(A * x_esrk - b) / norm_b;
            errs_esrk(iter_plot, repeat) = norm(x_esrk - x_hat) / norm_x_hat;
            errs_Bregman_esrk(iter_plot, repeat) = S_dist(xdual_esrk, x_hat, lambda) / varphi(x_hat);
            res_hb_inexact(iter_plot, repeat) = norm(A * x_hb_inexact - b) / norm_b;
            errs_hb_inexact(iter_plot, repeat) = norm(x_hb_inexact - x_hat) / norm_x_hat;
            errs_Bregman_hb_inexact(iter_plot, repeat) = S_dist(xdual_hb_inexact, x_hat, lambda) / varphi(x_hat);
            res_hb_double_inexact(iter_plot, repeat) = norm(A * x_hb_double_inexact - b) / norm_b;
            errs_hb_double_inexact(iter_plot, repeat) = norm(x_hb_double_inexact - x_hat) / norm_x_hat;
            errs_Bregman_hb_double_inexact(iter_plot, repeat) = S_dist(xdual_hb_double_inexact, x_hat, lambda) / varphi(x_hat);            
            res_hb_opt_beta(iter_plot, repeat) = norm(A * x_hb_opt_beta - b) / norm_b;
            errs_hb_opt_beta(iter_plot, repeat) = norm(x_hb_opt_beta - x_hat) / norm_x_hat;
            errs_Bregman_hb_opt_beta(iter_plot, repeat) = S_dist(xdual_hb_opt_beta, x_hat, lambda) / varphi(x_hat);
            for ii = 1:length(const_betas)
                res_hb(iter_plot, repeat, ii) = norm(A * x_hb(:,ii) - b) / norm_b;
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


%% plot errors

lightgray =   [0.8 0.8 0.8];
mediumgray =  [0.6 0.6 0.6];
lightred =    [1 0.9 0.9];
mediumred =   [1 0.6 0.6];
lightgreen =  [0.9 1 0.9];
mediumgreen = [0.6 1 0.6];
lightblue =   [0.9 0.9 1];
mediumblue =  [0.6 0.6 1];
lightmagenta =   [1 0.9 1];
mediummagenta =  [1 0.6 1];

% plot average residuals over iterations

mean_res_srk = mean(res_srk, 2);
mean_res_esrk = mean(res_esrk, 2);
mean_res_hb_inexact = mean(res_hb_inexact, 2);
mean_res_hb_double_inexact = mean(res_hb_double_inexact, 2);
mean_res_hb_opt_beta = mean(res_hb_opt_beta, 2);
mean_res_hb = zeros(length(num_iter_array), num_const_betas);
for ii = 1:num_const_betas
    mean_res_hb(:, ii) = mean(res_hb(:, :, ii), 2);
end

% sparse Kaczmarz
srk = semilogy(num_iter_array, mean_res_srk, 'Color', 'black');

hold on

% sparse Kaczmarz with exact step size
esrk = semilogy(num_iter_array, mean_res_esrk, 'Color', 'green');

% sparse Kaczmarz with inexact heavy ball momentum
%inexact_beta = semilogy(num_iter_array, mean_res_hb_inexact, 'Color', 'magenta');

% sparse Kaczmarz with both inexact step size and heavy ball momentum
double_inexact = semilogy(num_iter_array, mean_res_hb_double_inexact, 'Color', 'red');

% sparse Kaczmarz with minimum error heavy ball momentum 
min_err_beta = semilogy(num_iter_array, mean_res_hb_opt_beta, 'Color', 'blue');

% sparse Kaczmarz with constant heavy ball momentum

for ii = 1:num_const_betas
    color = const_betas(ii) * [0 1 1]; % interp. between black (beta=0) and cyan (beta=1)
    if ii == 1
        const_momentum = semilogy(num_iter_array, mean_res_hb(:, ii), 'Color', color);
    else
        semilogy(num_iter_array, mean_res_hb(:, ii), 'Color', color);
    end
end

legend([srk, esrk, double_inexact, min_err_beta, const_momentum], ...
        'SRK', 'ESRK', 'relaxed exact momentum', ...
        'exact momentum with SRK step size (SRKEM)', 'momentum with constant parameters');
        

title('Average relative residuals over iterations, mean over all repeats')

hold off



% plot quantiles against iterations

figure

display_names = {'SRK', 'ESRK', 'SRKREMS', 'SRKEM'}; 
arrs = {res_srk, res_esrk, res_hb_double_inexact, res_hb_opt_beta};

num_methods = length(arrs); 
line_colors = {'black', 'green', 'red', 'blue'}; 
minmax_colors = {lightgray, lightgreen, lightred, lightblue}; 
quant_colors = {mediumgray, mediumgreen, mediumred, mediumblue};

display_legend = true;
max_val_in_plot = 1e3;

[x_arrays, quantiles] = compute_and_plot_quantiles_in_logscale(num_iter_array, arrs, ...
                           num_methods, line_colors, display_names, ...
                           minmax_colors, quant_colors, display_legend, max_val_in_plot);

title('Relative residuals over iterations, with quantiles')




% display runtimes (min/mean/max)

min_rt_srk = min(rt_srk(end,:));
mean_rt_srk = mean(rt_srk(end,:));
max_rt_srk = max(rt_srk(end,:));

min_rt_esrk = min(rt_esrk(end,:));
mean_rt_esrk = mean(rt_esrk(end,:));
max_rt_esrk = max(rt_esrk(end,:));

min_rt_hb_double_inexact = min(rt_hb_double_inexact(end,:));
mean_rt_hb_double_inexact = mean(rt_hb_double_inexact(end,:));
max_rt_hb_double_inexact = max(rt_hb_double_inexact(end,:));

min_rt_hb_opt_beta = min(rt_hb_opt_beta(end,:));
mean_rt_hb_opt_beta = mean(rt_hb_opt_beta(end,:));
max_rt_hb_opt_beta = max(rt_hb_opt_beta(end,:));

fprintf('Runtime for SRK (min/median/max): %f / %f / %f \n', min_rt_srk, mean_rt_srk, max_rt_srk)
fprintf('Runtime for ESRK (min/median/max): %f / %f / %f \n', min_rt_esrk, mean_rt_esrk, max_rt_esrk)
fprintf('Runtime for SRKREMS (min/median/max): %f / %f / %f \n', min_rt_hb_double_inexact, mean_rt_hb_double_inexact, max_rt_hb_double_inexact)
fprintf('Runtime for SRKEM (min/median/max): %f / %f / %f \n', min_rt_hb_opt_beta, mean_rt_hb_opt_beta, max_rt_hb_opt_beta)
     
if writeout

    writeout_data_over_array_on_xaxis(...
    './randn_sparse/lambda=.1/res_over_iter_means.txt',...
    {'k', 'mean_res_srk', 'mean_res_esrk', 'mean_res_hb_double_inexact', 'mean_res_hb_opt_beta' ...
          'mean_res_hb_beta_0.1', 'mean_res_hb_beta_0.2', 'mean_res_hb_beta_0.3', 'mean_res_hb_beta_0.4', 'mean_res_hb_beta_0.5', ...
          'mean_res_hb_beta_0.6', 'mean_res_hb_beta_0.7', 'mean_res_hb_beta_0.8','mean_res_hb_beta_0.9', 'mean_res_hb_beta_0.96'}, ...
    num_iter_array, ...
    [mean_res_srk, mean_res_esrk, mean_res_hb_double_inexact, mean_res_hb_opt_beta, ...
    mean_res_hb(:, 1), mean_res_hb(:, 2), mean_res_hb(:, 3), mean_res_hb(:, 4), mean_res_hb(:, 5), ...
    mean_res_hb(:, 6)]) 

    writeout_data_over_array_on_xaxis(...
        './randn_sparse/lambda=.1/res_over_iter_quantiles.txt',...
        {'k', 'min_res_srk', 'min_res_esrk', 'min_res_hb_double_inexact', 'min_res_hb_opt_beta', ...
              'max_res_srk', 'max_res_esrk', 'max_res_hb_double_inexact', 'max_res_hb_opt_beta', ...
              'median_res_srk', 'median_res_esrk', 'median_res_hb_double_inexact', 'median_res_hb_opt_beta', ...
              'quant25_res_srk', 'quant25_res_esrk', 'quant25_res_hb_double_inexact', 'quant25_res_hb_opt_beta', ...
              'quant75_res_srk', 'quant75_res_esrk', 'quant75_res_hb_double_inexact', 'quant75_res_hb_opt_beta'}, ...
        num_iter_array, ...
        [quantiles{1}{1}, quantiles{1}{2}, quantiles{1}{3}, quantiles{1}{4}, ...
        quantiles{2}{1}, quantiles{2}{2}, quantiles{2}{3}, quantiles{2}{4}, ...
        quantiles{3}{1}, quantiles{3}{2}, quantiles{3}{3}, quantiles{3}{4}, ...
        quantiles{4}{1}, quantiles{4}{2}, quantiles{4}{3}, quantiles{4}{4}, ...
        quantiles{5}{1}, quantiles{5}{2}, quantiles{5}{3}, quantiles{5}{4}]) 

end
