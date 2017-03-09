function [w, iter_cost] = BGD(X, y, a, lambda, max_iter)

X = [X, ones(size(X, 1), 1)];
w = randn(size(X, 2), 1);

iter_cost = zeros(max_iter, 1);
for i = 1:max_iter
    g = gradient(X, w, y, lambda);
    w = w - a*g;
    iter_cost(i) = cost(X, w, y, lambda);
end
