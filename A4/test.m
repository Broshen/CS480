clear; clc


tt = 1500;
T = [-1:2/tt:1]';

tt = length(T);
n = 10;


avg = @(x) mean(x);

kappa1 = @(s, t) exp(-(s-t)^2/(2*0.1));
kappa2 = @(s, t) exp(-(s-t)^2/(2*1));
kappa3 = @(s, t) exp(-(s-t)^2/(2*10));


d2kappa1 = @(s, t) kappa1(s,t)*(1-((s-t)^2)/0.1)/0.1;
d2kappa2 = @(s, t) kappa2(s,t)*(1-((s-t)^2));
d2kappa3 = @(s, t) kappa3(s,t)*(1-((s-t)^2)/10)/10;

gp(T, n, avg, kappa1, '0.1.png')
gp(T, n, avg, kappa2, '1.png')
gp(T, n, avg, kappa3, '10.png')


gp(T, n, avg, d2kappa1, 'd0.1.png')
gp(T, n, avg, d2kappa2, 'd1.png')
gp(T, n, avg, d2kappa3, 'd10.png')