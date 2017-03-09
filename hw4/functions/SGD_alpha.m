function [w, iter_cost] = SGD_alpha(X, y, a, lambda, max_iter)

X = [X, ones(size(X, 1), 1)];
w = zeros(size(X, 2), 1);

iter_cost = zeros(max_iter, 1);
for i = 1:max_iter
    if mod(i, 1000)==0
        fprintf('iter: %d\n', i);
    end
    idx = ceil(rand()*size(X, 1));
%     g = gradient(X(idx, :), w, y(idx), lambda);
    Xi = X(idx, :);
    yi = y(idx);
    s = 1/(1+exp(Xi*w));
    g = -Xi' * (yi-s) + 2*lambda*w;
    w = w - a/i*g;
    iter_cost(i) = -(y'*log(sigmoid(X, w)+eps) + (1-y)'*log(1-sigmoid(X, w)+eps)) + lambda*(w'*w);
end
