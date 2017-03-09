% This is the main script of hw4 question 4.
% To run the code, put data.mat in the same directory with this script.
%
% Written by Yicheng Chen
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
addpath('./functions')
load('data.mat')

miu = mean(X, 1);
sigma = std(X, 1);
repmat_num = size(X)./size(miu);
X = (X-repmat(miu, repmat_num))./repmat(sigma, repmat_num);
repmat_num = size(X_test)./size(miu);
X_test = (X_test-repmat(miu, repmat_num))./repmat(sigma, repmat_num);

%% build validation set
idx = randsample(1:size(X, 1), size(X, 1));
X = X(idx, :);
y = y(idx, :);
X_train = X(1:5000, :);
y_train = y(1:5000, :);
X_valid = X(5001:end, :);
y_valid = y(5001:end, :);

%% BGD
a = 1e-3;
lambda = 1e0;
max_iter = 200;
[w, iter_cost] = BGD(X, y, a, lambda, max_iter);
figure;
plot(1:max_iter, iter_cost);
iter_cost(end)

%% SGD
a = 1e-3;
lambda = 1e0;
max_iter = 10000;
[w, iter_cost] = SGD(X, y, a, lambda, max_iter);
figure;
plot(1:max_iter, iter_cost);
iter_cost(end)


%% SGD with changing learning rate
a = 1e-1;
lambda = 1e0;
max_iter = 10000;
[w, iter_cost] = SGD_alpha(X, y, a, lambda, max_iter);
figure;
plot(1:max_iter, iter_cost);
iter_cost(end)


%% tune lambda
a = 1e-2;
max_iter = 100;
lambda = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0, 1, 10];
% lambda = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 2];
all_accuracy = zeros(size(lambda));
for i = 1:length(lambda)
    [w, iter_cost] = BGD(X_train, y_train, a, lambda(i), max_iter);
    yp = classifier(X_valid, w);
    all_accuracy(i) = sum(yp == y_valid) / 1000;
end

figure;
semilogx(lambda, all_accuracy, 'x');

%% Kaggle
a = 1e-2;
lambda = 1e0;
max_iter = 5000;
[w, iter_cost] = BGD(X, y, a, lambda, max_iter);
plot(1:max_iter, iter_cost);
iter_cost(end)
y_test = classifier(X_test, w);
data = 0:size(y_test, 1)-1;
data = [data' y_test];
csvwrite_with_headers('wine_predict.csv', data, {'Id', 'Category'});

%% test
p0 = csvread('wine_predict_0.csv');
