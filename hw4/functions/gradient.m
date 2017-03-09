function g = gradient(X, w, y, lambda)

g = -X' * (y-sigmoid(X, w)) + 2*lambda*w;
