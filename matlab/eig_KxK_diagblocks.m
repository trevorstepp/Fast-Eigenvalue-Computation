function [eigvals, eigvecs] = eig_KxK_diagblocks(Blocks)

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