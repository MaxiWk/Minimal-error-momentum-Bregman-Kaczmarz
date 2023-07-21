% computes a common timegrid (uniform over all repeats) for an array
% 'runtime_over_iter' of size floor(maxiter/iter_save) x num_repeats
% (just uniform such that it fits every column)
function timegrid = compute_timegrid(runtime_over_iter, num_timegrid_points)

    num_timegrid_points = min(num_timegrid_points, size(runtime_over_iter, 1) );

    max_timegrid = min(runtime_over_iter(end, :));

    gridsize = max_timegrid / num_timegrid_points;

    timegrid = (gridsize : gridsize : max_timegrid)';

end




