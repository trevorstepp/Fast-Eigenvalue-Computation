function r = relerr(a,b)
    a = sort(a,'ComparisonMethod','abs');
    b = sort(b,'ComparisonMethod','abs');
    r = norm(a-b)/max(1e-16,norm(b));
end