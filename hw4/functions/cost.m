function c = cost(X, w, y, lambda)

c = -(y'*log(sigmoid(X, w)+eps) + (1-y)'*log(1-sigmoid(X, w)+eps)) + lambda*(w'*w);