function gp(T, n, avg, kappa, graphname)

rng(480)  % set random number generator seed for consistent results
m = avg(T);
t = length(T);
K = zeros(t);
if isa(kappa, 'function_handle')
    for i = 1:t
        for j = i:t
            K(i,j) = kappa(T(i), T(j));
        end
    end
    K = K + triu(K,1)';
else
    K = kappa;
end

% add small identity to prevent numerical issue
L = chol(K+1e-5*eye(t), 'lower');

X = zeros(t, n);
plot(T, m, '--b', 'LineWidth', 4);
hold on
for ii = 1:n
    x = L * randn(t, 1);
    x = m + x;
    X(:,ii) = x;
    if length(x) == 1
        plot(T, x, 'o');
    else
        plot(T, x);
    end
    hold on
end

hold off

saveas(gcf, graphname);

