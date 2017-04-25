clear;
% read data
train = load('data/joke_data/joke_train.mat');
train = train.train;
%%
[val_i, val_j, val_s] = textread('data/joke_data/validation.txt', '%d,%d,%d');
[que_id, que_i, que_j] = textread('data/joke_data/query.txt', '%d,%d,%d');

%% 1. zero-filled
all_k = [2, 5, 10, 20];
all_accuracy = zeros(1, length(all_k));
all_MSE = all_accuracy;
for i = 1:length(all_k)
    k = all_k(i);
    [U, V] = latent_factor_zf(train, k);
    all_accuracy(i) = validation_accuracy(U, V, val_i, val_j, val_s);
    Rzf = train; Rzf(isnan(Rzf)) = 0;
    all_MSE(i) = immse(Rzf, U*V);
end
fprintf('Accuracy on validation set: ')
disp(all_accuracy)
figure; plot(all_k, all_MSE, 'x-')
title('MSE')
xlabel('k')
ylabel('MSE')
%%
k = 2;
all_lam = [1e10, 1e8, 1e6, 1e4, 1e2, 1e0];
all_p = zeros(1, length(all_lam));
for i =1:length(all_lam)
    lam = all_lam(i);
    [U, V] = latent_factor_reg(train, k, lam);
    all_p(i) = validation_accuracy(U, V, val_i, val_j, val_s);
end
disp(all_p)

%%
tic;
all_k = [2, 5, 10, 20];
lam = 1e1;
all_p = zeros(1, length(all_k));
all_MSE = all_p;
for i =1:length(all_k)                                                     
    k = all_k(i);
    [U, V] = latent_factor_reg(train, k, lam);
    all_p(i) = validation_accuracy(U, V, val_i, val_j, val_s);
    Rzf = train;
    Rzf(isnan(Rzf)) = 0;
    Rapprox = U*V;
    diff = (Rapprox-Rzf).^2;
    all_MSE(i) = mean(diff(:));
end
toc;
disp(all_p)
%%
Rzf = train;
Rzf(isnan(Rzf)) = 0;
Rapprox = U*V;
diff = (Rapprox-Rzf).^2;
MSE = mean(diff(:))

%% prediction
p = predict(U, V, que_i, que_j);
A = [que_id'; p'];
f = fopen('Kaggle_submission.csv', 'w');
fprintf(f, 'Id,Category\n');
fprintf(f, '%d,%d\n', A);
fclose(f);
disp('Writen to csv');
