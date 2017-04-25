function p = predict(U, V, i, j)

p = zeros(length(i), 1);
for ii = 1:length(i)
    pp = U(i(ii), :) * V(:, j(ii));
    if pp > 0
        p(ii) = 1;
    else
        p(ii) = 0;
    end
end
