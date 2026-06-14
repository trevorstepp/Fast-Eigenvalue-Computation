T = readtable('matlab_large_n_timings.csv');

% --- Plot ---
figure;
semilogy(T.n,T.block_time,'o-', ...
    'LineWidth',1.5);

xlabel('discretization size (n)'); 
ylabel('run time (s)'); 
title(sprintf('All eigenpairs runtime (K=%d)',K)); 
grid on;

legend('Block (large n)', ...
    'Location','northwest');

savefig("test.fig")