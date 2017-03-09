function y = sigmoid(X, w)

y = 1./(1+exp(-X*w));