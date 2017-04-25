function [U, V] = latent_factor_zf(R, k)

Rzf = R;
Rzf(isnan(Rzf)) = 0;
[U, S, V] = svd(Rzf,'econ');

U = U(:, 1:k) * sqrt(S(1:k, 1:k));
V = sqrt(S(1:k, 1:k)) * V(:, 1:k)';