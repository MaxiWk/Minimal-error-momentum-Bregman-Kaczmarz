   function [mins, maxs, medians, quant25s, quant75s] = compute_minmax_median_quantiles(arr)
        if size(arr, 2) == 1   % if only one repeat, nothing to compute 
           [mins, maxs, medians, quant25s,quant75s] = deal(arr);
        else
           mins = min(arr, [], 2);
           maxs = max(arr, [], 2);
           medians = median(arr,2);
           quant25s = quantile(arr,0.25,2); 
           quant75s = quantile(arr,0.75,2);
        end
    end