function [U, V] = latent_factor_reg(R, k, lam)
    U = randn(size(R, 1), k);
    V = randn(k, size(R, 2));
    idx = ~isnan(R);
    while 1
        U1 = zeros(size(U));
        V1 = zeros(size(V));
        % update U
        parfor i = 1:size(U, 1)
            Vsi = V(:, idx(i, :));
            Rsi = R(i, idx(i, :))';
            ui = (Vsi*Vsi'+lam*eye(k))\(Vsi*Rsi);
            U1(i, :) = ui';
        end
        % update V
        parfor j = 1:size(V, 2)
            Usj = U1(idx(:, j), :);
            Rsj = R(idx(:, j), j);
            vj = (Usj'*Usj+lam*eye(k))\(Usj'*Rsj);
            V1(:, j) = vj;
        end
        diffU = abs(U1-U);
        diffV = abs(V1-V);
        fprintf('diffU:%f, diffV:%f\n', mean(diffU(:)), mean(diffV(:)))
        if mean(diffU(:))<1e-3 && mean(diffU(:))<1e-3
            break
        else
            U = U1;
            V = V1;
        end
    end
end


