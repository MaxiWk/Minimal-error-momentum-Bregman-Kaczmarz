% Runtime postprocessing: 
% for each method, compute a timegrid which is uniform over all repeats
% and round data correspondingly

% Input: 
% timegrid: array of size num_timegrid_points x 1
% rt: matrix of size "floor(maxiter/iter_save) x num_repeats"
% data: cell array of matrices of size "floor(maxiter/iter_save) x num_repeats"

% Output: 
% data_rounded_to_timegrid: cell array of matrices of size
%       num_timegrid_points x num_repeats 
% t_idx_with_t_on_grid: matrix of same size, contains corresponding indices

function [t_idx_with_t_on_grid, data_rounded_to_timegrid] = round_to_timegrid(timegrid, rt, data, num_repeats)

    num_timegrid_points = size(timegrid, 1);

    [t_idx_with_t_on_grid, data_rounded_to_timegrid] = deal( zeros( num_timegrid_points, num_repeats ) );

    for repeat = 1:num_repeats 

        jj = 1;
        for ii = 1:size(rt, 1) 
            if jj == num_timegrid_points
                break
            end
            if rt(ii, repeat) < timegrid(jj) + eps
                t_idx_with_t_on_grid(jj, repeat) = ii;
            else 
                t_idx_with_t_on_grid(jj+1, repeat) = ii;
                jj = jj + 1;
            end
        end

        data_rounded_to_timegrid(:, repeat) = data( t_idx_with_t_on_grid(:, repeat), repeat );

    end

end