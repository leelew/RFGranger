function index = index
    index.R2 = @determination_coefficient;
    index.MSE = @mean_squared_error;
    index.RMSE = @root_mean_squared_error;
    index.MAE = @mean_average_error;
    index.AIC = @Akaike_information_criterion;
    index.KLDiv = @KL_divergence;
end

function R2 = determination_coefficient(x,y)

R2 = 1 - (sum((y-x).^2)./sum((y-mean(y)).^2));

end

function MSE = mean_squared_error(x,y)

MSE = nansum((y-x).^2)/length(y);

end

function RMSE = root_mean_squared_error(x,y)

RMSE = mean_squared_error(x,y).^0.5;

end

function MAE = mean_average_error(x,y)

MAE = sum(abs(x-y))/numel(y);

end

function AIC = Akaike_information_criterion(N,dev,aic_penalty)

AIC = aic_penalty*(N+1)-2*(-dev/2);

end

function dist = KL_divergence(P,Q)

%  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
%  distributions
%  P and Q  are automatically normalised to have the sum of one on rows
%  have the length of one at each 
%  P =  n x nbins
%  Q =  1 x nbins or n x nbins(one to one)
%  dist = n x 1


if size(P,2)~=size(Q,2)
    error('the number of columns in P and Q should be the same');
end

if sum(~isfinite(P(:))) + sum(~isfinite(Q(:)))
   error('the inputs contain non-finite values!') 
end

% normalizing the P and Q
if size(Q,1)==1
    
    Q = Q ./sum(Q);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    temp =  P.*log(P./repmat(Q,[size(P,1) 1]));
    temp(isnan(temp))=0; % resolving the case when P(i)==0
    dist = sum(temp,2);
        
elseif size(Q,1)==size(P,1)
    
    Q = Q ./repmat(sum(Q,2),[1 size(Q,2)]);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    temp =  P.*log(P./Q);
    temp(isnan(temp))=0; % resolving the case when P(i)==0
    dist = sum(temp,2);
    
end
end



