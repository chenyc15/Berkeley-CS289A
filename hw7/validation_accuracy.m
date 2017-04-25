function p = validation_accuracy(U, V, i, j, s)

n = length(s);
nCorrect = 0;
for ii = 1:n
    predict = U(i(ii), :) * V(:, j(ii));
    if predict * (s(ii)-0.5) > 0
        nCorrect = nCorrect + 1;
    end
end

p = nCorrect / n;