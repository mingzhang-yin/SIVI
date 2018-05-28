function y = logOnePlusExp(x)
dex = x<0;
y = zeros(size(x));
%y(dex) = log(1+exp(x(dex)));
%y(~dex) = x(~dex) + log(1+exp(-x(~dex)));
y(dex) = log1p(exp(x(dex)));
y(~dex) = x(~dex) + log1p(exp(-x(~dex)));
