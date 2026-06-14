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