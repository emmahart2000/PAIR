function err = avg_rel_error(y_pred, y_true)
    numer = sqrt(sum((y_pred-y_true).^2,1));
    denom = sqrt(sum((y_true).^2,1));
    err = mean(numer./denom);
end
