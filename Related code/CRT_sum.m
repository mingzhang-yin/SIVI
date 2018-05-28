
function Lsum= CRT_sum(x,r,IsMexOK)
if nargin<3
    IsMexOK = false;
end
if ~IsMexOK
    Lsum=0;
    RND = r./(r+(0:(max(x)-1)));
    for i=1:length(x);
        if x(i)>0
            Lsum = Lsum + sum(rand(1,x(i))<=RND(1:x(i)));
        end
    end
else
    %Lsum = CRT_sum_mex(x,r,rand(sum(x),1),max(x));
    Lsum = CRT_sum_mex(x,r);
end
