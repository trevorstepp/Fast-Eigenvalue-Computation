clear; clc; rng(42);

K  = 3;
ns = [100 250 500 750 1000 1500 2000 3000 4500 6000 10000];

t_block_all = NaN(size(ns));

fprintf('===== Benchmark: %dx%d block matrix =====\n',K,K);

for s = 1:numel(ns)
    n = ns(s);
    fprintf('Running n = %d\n',n)

    % --- Store ONLY diagonals (fix #1) ---
    Blocks = cell(K,K);
    for p = 1:K
        for q = 1:K
            Blocks{p,q} = randn(n,1);   % vector, not diag matrix
        end
    end

    f_block = @() eig_KxK_diagblocks_clean(Blocks);
    t_block_all(s) = timeit(f_block);
end

T_all = table(ns(:),t_block_all(:),...
    'VariableNames',{'n','Time_BlockAll'});
disp(T_all);

figure;
semilogy(ns,t_block_all,'o-','LineWidth',1.5);
xlabel('n'); ylabel('time (s)');
title('Cleaned implementation'); grid on;

% ============================================================
function [eigvals, eigvecs] = eig_KxK_diagblocks_clean(Blocks)

    K = size(Blocks,1);
    n = numel(Blocks{1,1});
    N = K*n;

    eigvals = zeros(N,1);

    % --- sparse storage (fix #2) ---
    nnz_est = K*K*n;
    I = zeros(nnz_est,1);
    J = zeros(nnz_est,1);
    Vals = zeros(nnz_est,1);

    Mi = zeros(K,K);

    col = 1;
    ptr = 1;

    for i = 1:n
        % build Mi
        for p = 1:K
            for q = 1:K
                Mi(p,q) = Blocks{p,q}(i);
            end
        end

        [V,L] = eig(Mi,'vector');
        eigvals(col:col+K-1) = L;

        % fill sparse eigvecs
        for j = 1:K
            for p = 1:K
                I(ptr) = (p-1)*n + i;
                J(ptr) = col + j - 1;
                Vals(ptr) = V(p,j);
                ptr = ptr + 1;
            end
        end

        col = col + K;
    end

    eigvecs = sparse(I,J,Vals,N,N);
end
%{
% pick one size to validate
n = 500;
K = 3;

Blocks = cell(K,K);
for p = 1:K
    for q = 1:K
        Blocks{p,q} = randn(n,1);
    end
end

[eigvals, eigvecs] = eig_KxK_diagblocks_clean(Blocks);

[max_res, mean_res] = eig_residuals_diagblocks(Blocks, eigvals, eigvecs);

fprintf('Max residual: %.2e\n', max_res);
fprintf('Mean residual: %.2e\n', mean_res);


function [max_res, mean_res] = eig_residuals_diagblocks(Blocks, eigvals, eigvecs)

    K = size(Blocks,1);
    n = numel(Blocks{1,1});
    N = K*n;

    residuals = zeros(N,1);

    for k = 1:N
        v = eigvecs(:,k);
        lambda = eigvals(k);

        rnorm_sq = 0;

        for i = 1:n
            for p = 1:K
                idx_p = (p-1)*n + i;

                Mv_pi = 0;
                for q = 1:K
                    idx_q = (q-1)*n + i;
                    Mv_pi = Mv_pi + Blocks{p,q}(i) * v(idx_q);
                end

                r = Mv_pi - lambda * v(idx_p);
                rnorm_sq = rnorm_sq + r*r;
            end
        end

        residuals(k) = sqrt(rnorm_sq);
    end

    max_res = max(residuals);
    mean_res = mean(residuals);
end
%}