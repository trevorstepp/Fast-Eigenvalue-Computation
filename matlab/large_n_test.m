%{
The test in this file is meant to show how runtime scales with even larger 
n than those shown in the main file (where we compare with the built-in 
method and calculate residuals). We do not check residuals as it would 
significantly increase runtime with such large n values. Here we are only 
interested in obtaining the runtime so we can make a log-log plot to obtain 
the slope and see if it matches the expected O(nK^3). Since we hold K 
constant, we expect a slope of 1 to match O(n).
%}
clear; clc; rng(42);

K  = 3;
ns = [100 250 500 750 1000 1500 2000 3000 4500 6000 10000];

t_block_all   = NaN(size(ns));
max_residuals = NaN(size(ns));

fprintf('===== Benchmark: %dx%d block matrix =====\n',K,K);

for s = 1:numel(ns)
    n = ns(s);
    fprintf('Running n = %d\n',n)

    % --- Store ONLY diagonals (fix #1) ---
    Blocks = cell(K,K);
    for p = 1:K
        for q = 1:K
            Blocks{p,q} = randn(n,1);
        end
    end

    f_block = @() eig_KxK_diagblocks(Blocks);
    t_block_all(s) = timeit(f_block);

    % timeit does not return values of the function call, so
    % need an extra call if you want the eigenvalues and eigenvectors
    %[eigvals, eigvecs] = eig_KxK_diagblocks(Blocks);
    %[max_res, mean_res] = eig_residuals_diagblocks(Blocks, eigvals, eigvecs);
    
    %fprintf('Max residual: %.2e\n', max_res);
    %fprintf('Mean residual: %.2e\n', mean_res);
end

% storage in .csv for log-log plot to view slope
T = table( ...
    repmat(K, numel(ns), 1), ...
    ns(:), ...
    t_block_all(:), ...
    'VariableNames', {'K','n','block_time'} ...
);
writetable(T, 'matlab_large_n_timings.csv');

T_all = table(ns(:),t_block_all(:),...
    'VariableNames',{'n','Time_BlockAll'});
disp(T_all);

figure;
semilogy(ns,t_block_all,'o-','LineWidth',1.5);
xlabel('n'); ylabel('time (s)');
title('Cleaned implementation'); grid on;