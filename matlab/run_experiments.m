clear; clc; rng(42);

% --- Controls ---
K        = 3;                  % number of diagonal blocks in each dimension
ns       = [100 250 500 750 1000 1500 2000];   % adjust for your patience

% --- Storage ---
t_block_all   = NaN(size(ns));
t_eig_full    = NaN(size(ns));
max_residuals = NaN(size(ns));

err_all       = NaN(size(ns));

fprintf('===== Benchmark: %dx%d block matrix =====\n',K,K);

for s = 1:numel(ns)
    n = ns(s); N = K*n;
    fprintf('Running n = %d\n',n)

    % --- Build K×K diagonal blocks ---
    Blocks = cell(K,K);
    for p = 1:K
        for q = 1:K
            Blocks{p,q} = randn(n,1); % each block diagonal
        end
    end

    % --- Assemble sparse M ---
    M = sparse(N,N);
    for p = 1:K
        for q = 1:K
            rows = (p-1)*n + (1:n);
            cols = (q-1)*n + (1:n);
            M(rows,cols) = diag(Blocks{p,q});
        end
    end
    
    % --- Block ---
    f_block = @() eig_KxK_diagblocks(Blocks);
    t_block_all(s) = timeit(f_block);

    % timeit does not return values of the function call, so
    % need an extra call if you want the eigenvalues and eigenvectors
    [eigvals, eigvecs] = eig_KxK_diagblocks(Blocks);
    [max_res, mean_res] = eig_residuals_diagblocks(Blocks, eigvals, eigvecs);

    fprintf('Max residual: %.2e\n', max_res);
    fprintf('Mean residual: %.2e\n', mean_res);
    max_residuals(s) = max_res;
    
    % --- eig --- 
    Md = full(M);
    f_full = @() eig(Md);
    t_eig_full(s) = timeit(f_full);

    [V, D] = eig(Md);
    lam_eig_full = diag(D);
    err_all(s) = relerr(lam_eig_full, eigvals);
end

% storage in .csv for plotting runtime
T = table( ...
    repmat(K, numel(ns), 1), ...
    ns(:), ...
    t_block_all(:), ...
    t_eig_full(:), ...
    max_residuals(:), ...
    'VariableNames', {'K','n','block_time','dense_time','max_residual'} ...
);

writetable(T, 'matlab_timings.csv');

% --- Plot ---
figure;
semilogy(ns,t_block_all,'o-',ns,t_eig_full,'s-','LineWidth',1.5);
xlabel('discretization size (n)'); ylabel('run time (s)'); title(sprintf('All eigenpairs runtime (K=%d)',K)); grid on;
legend('Block(all)','eig(full)','Location','northwest');

% --- Table ---
T_all = table(ns(:),t_block_all(:),t_eig_full(:),err_all(:),...
    'VariableNames',{'n','Time_BlockAll','Time_EigFull','RelErr'});
disp('--- All eigenpairs ---'); disp(T_all);