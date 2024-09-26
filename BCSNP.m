# BCSNP
% Bayesian Compressive Sensing Using Normal Product Priors
function [ est_x,est_s] = BCSNP(y,N,M,A,maxiter)
% y is the observation
% epslon is the noise standard deviation, here could be regarded as a
% tunning parameter
% This file eliminate the ill posed problem in noieless case+prunning
%%
    kappa=ones(N,1);% The standard deviation of a
    gamma=ones(N,1);% The standard deviation of b
    % (kappa^2+gamma^2)^2/4 determine the standard variance of gaussian
    %  product

    mean_a=ones(N,1);
    mean_b=ones(N,1);
    converged=false;
    iter=0;
    x_new=mean_a.*mean_b;
    index=1:N;
    Num=N;
while ~converged
    
            x_old=x_new;
    
            %prunning:
%             mask=(x_new >1e-5);
%             Num=sum(mask);
%             index=index(mask);
%             A=A(:,mask);
%             mean_a=mean_a(mask);
%             kappa=kappa(mask);
%             gamma=gamma(mask); 
   
            %%%%%%%%% update b %%%%%%%%
            var_b=(eye(Num)-diag(gamma)*pinv(A*diag(mean_a)*diag(gamma))*A*diag(mean_a))*diag(gamma.^2);
            mean_b=diag(gamma)*pinv(A*diag(mean_a)*diag(gamma))*y;
       
            %%%%%%%%%% update a%%%%%%%%
             var_a=(eye(Num)-diag(kappa)*pinv(A*diag(mean_b)*diag(kappa))*A*diag(mean_b))*diag(kappa.^2);
             mean_a=diag(kappa)*pinv(A*diag(mean_b)*diag(kappa))*y;
            
%             %%%%%%%%% update gamma %%%%%%
             kappa=sqrt(mean_a.^2+diag(var_a));
            
            %%%%%%%% update kappa %%%%%%%%
             gamma=sqrt(mean_b.^2+diag(var_b));
            
            
            x_new=mean_a.*mean_b;
            if  norm(x_new-x_old)/norm(x_old)<1e-9||iter>=maxiter
                converged=true;
            end
            %fprintf('The number of iteration was %d\n',iter);
            iter=iter+1;
end
%             est_x=zeros(N,1);
%             est_x(index)=x_new;
            est_x=x_new;
            est_s=(abs(x_new)>1e-3);

end
