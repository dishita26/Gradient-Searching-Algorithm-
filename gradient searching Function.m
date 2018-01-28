function [J,a] = grad_descent(trainset, a,b, eta, theta)
n = size(trainset,1);
k = 0;
stop = 0;
kmax = 10000;
J = [];

while((stop==0 && k<kmax))
  k = k+1;
  for i = 1:length(trainset)
      w_delta = (eta)*(trainset(i,:))* ((b(i,:) - trainset(i,:)*a'));
      
      %Stop Criterion
      if(norm(w_delta) > theta)
          a = a +w_delta;
      else
          stop = 1;
          break;
      end
       J(k) = norm(trainset*a' - b);
      disp([k]);
  end
end
