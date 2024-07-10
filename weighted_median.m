function wm = weighted_median(values, weights)
    % Ensure that the values and weights are column vectors
    values = values(:);
    weights = weights(:);

    % Sort the values and weights based on values
    [sorted_values, sort_idx] = sort(values);
    sorted_weights = weights(sort_idx);

    % Compute the cumulative weights
    cumulative_weights = cumsum(sorted_weights);

    % Find the weighted median
    total_weight = sum(weights);
    median_idx = find(cumulative_weights >= total_weight / 2, 1, 'first');
    wm = sorted_values(median_idx);
end