clear;
addpath('./functions')
X = [0 3;1 3;0 1;1 1;];
y = [1;1;0;0];
X = [X, ones(4, 1)];
l = 0.07;
w0 = [-2;1;0];

s0 = sigmoid(X, w0);

w1 = w0 + inv(2*l*ones(3)+X'*diag(s0)*X)*(2*l*w0-X'*(y-s0));
s1 = sigmoid(X, w1);
w2 = w1 + inv(2*l*ones(3)+X'*diag(s1)*X)*(2*l*w1-X'*(y-s1));