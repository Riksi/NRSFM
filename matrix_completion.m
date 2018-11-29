% >> First calls matrix_completion(Wtmp,H,4,2,4)
% X = Wtmp (rescale W_k)
% W = H
% rfinal = 4
% mode = 2
% rinit = 4
% So does the loop the first time have just 1 iteration?
function [Z,U,V]=ALM_nuclear(X,W,rfinal,mode,rinit)
Z=[];
U=[];
V=[];
if nargin<5
    %for rank continuation to avoid local minima
    rinit=rfinal+3;
end
%factrize X:mXn into U:mXr and V:rXn
epsilon=0.1;
rho=10^(-5);
lambda=10^(-3);
mu=1.05;
[m,n]=size(X);
for r=rinit:-1:rfinal
    if isempty(Z)
        % >> For the first iteration randomly initialise U, V
        % from N(0,1)
        U=random('normal',0,1,[m,r]);
        V=random('normal',0,1,[n,r]);
        % >> Z has the same values as X when points are not missing
        % but where points are missing they are randomly initialised
        % from N(0,1)
        Z=(W>0).*X+(W==0).*random('normal',0,1,size(X));
    else
        % >> In subsequent iterations, U, V initialised from decomposing
        % the present value of Z
        % >> svds returns top r singular values
        % looks like it will return fewer if there are
        % fewer than r SVs, in which case you
        [U1,S,V1]=svds(Z,r);
        if size(S,1)<r
            Z=[];
            U=[];
            V=[];
            return; % >> Not sure what the behaviour will be if you return [] for Z,U,V
        end
        U=U1*sparse([1:r],[1:r],sqrt(diag(S)));
        V=V1*sparse([1:r],[1:r],sqrt(diag(S)));
    end
    max_iter=1000000;
    Y=rho*(Z-U*V');
    iter2=0;
    
    while 1
        iter1=0;
        iter2=iter2+1;
        progress('L2 bilinear factorization  iter',iter2,1000);
        % fprintf('outer iter %d \n',iter2);
        while 1
            iter1=iter1+1;
            % >> This is just solving for U from Y as defined above
            % and added small diagonal noise (for inversion I think)
            Unew=(rho*Z+Y)*V*(rho*(V'*V)+lambda*eye(r))^(-1);
            %                 if mod(iter1,100)==0
            %                 figure(1);clf;
            %                 subplot(1,2,1)
            %                 imagesc(Unew);
            %                 subplot(1,2,2)
            %                 imagesc(Vnew);s
            %                 end
            % >> Similar to Unew except that this time, we are using
            % the new value of U (next time we get Unew the V will be Vnew)
            Vnew=(rho*Z+Y)'*Unew*(rho*(Unew'*Unew)+lambda*eye(r))^(-1);
            if ~isempty(find(isnan(Unew)))
                error('nan update');
            end
            if ~isempty(find(isnan(Vnew)))
                error('nan update');
            end
            % >> Need to understand these updates
            if mode==1
                Znew=update_Z_L1(W,Unew,Vnew,Y,rho);
            else
                
                Znew=update_Z_L2(W,Unew,Vnew,Y,rho);
            end
            % fprintf('d=%d \n',norm(Znew-Z,'fro'))
            if iter1>max_iter ||norm(Znew-Z,'fro')<epsilon
                U=Unew;
                V=Vnew;
                Z=Znew;
                break;
            end
            U=Unew;
            V=Vnew;
            Z=Znew;
        end
        Y=Y+rho*(Z-U*V');
        rho=min(rho*mu,10^20);
        %  fprintf('rho=%0.8f',rho);
        Znew=U*V';
        if ~isempty(find(isnan(Znew)))
            error('nan update');
        end
        if iter2>max_iter ||norm(Z-Znew,'fro')<epsilon
            Z=Znew;
            break;
        end
        Z=Znew;
    end
    
    Z=U*V';
end

    function Z=update_Z_L2(W,U,V,Y,rho)
        Z=W.*((1/(2+rho))*(2*X+rho*(U*V'-rho^(-1)*Y)))+(W==0).*(U*V'-rho^(-1)*Y);
        if ~isempty(find(isnan(Z)))
            error('nan update');
        end
    end

    function Z=update_Z_L1(W,U,V,Y,rho)
        Z=W.*(X-max(0,X-U*V'+rho^(-1)*Y-rho^(-1)))+...
            (W==0).*(U*V'-rho^(-1)*Y);
    end

end