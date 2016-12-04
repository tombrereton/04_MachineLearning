function [cluster_means,ClusterIndex] = kmeans_cluster3(X,K)

% Code derived from Dr Simon Rogers
% revised code by Dr Kashif Rajpoot
% This is a simple implementation. For computer based assignment 3,
% you can try using this code or MATLAB's implementation (see help kmeans).

% plot data
figure(1);hold off
plot(X(:,1),X(:,2),'ko');
title('Original data');
pause

% Randomly initialise the means
cluster_means = rand(K,size(X,2))*max(X(:))-max(X(:))/2;

% Iteratively update the means and assignments
N = size(X,1);
ClusterIndex = zeros(N,1);
di = zeros(N,K);

% iterative process for objects assignment and means update
converged = 0;
iter=1;
while ~converged
    % find the distance estimate
    for k = 1:K
        % squared Euclidean distance
        di(:,k) = sum((X - repmat(cluster_means(k,:),N,1)).^2,2);
    end
    
    % assign objects to nearest cluster
    oldIndex = ClusterIndex;
    [notUsed,ClusterIndex] = min(di,[],2);

    if sum(oldIndex~=ClusterIndex) == 0 % any changes in assignments?
        converged = 1;
    end
    
    % Update means
    for k = 1:K   
        if sum(ClusterIndex==k)==0 % empty cluster?
            % This cluster is empty, randomise it
            cluster_means(k,:) = rand(1,size(X,2))*max(X(:))-max(X(:))/2;
        else
            cluster_means(k,:) = mean(X(ClusterIndex==k,:),1);
        end
    end
    
    % Plot the assigned data
    cols = {'r','g','b','k','m','c','y'};
    figure(1); hold off
    for k = 1:K
        plot(X(ClusterIndex==k,1),X(ClusterIndex==k,2),...
            'ko','markerfacecolor',cols{k});
        hold on
    end
    % Plot the means
    figure(1)
    for k = 1:K
        plot(cluster_means(k,1),cluster_means(k,2),'ks','markersize',15,...
            'markerfacecolor',cols{k});
    end
    title(sprintf('K = %d, after %d iteration(s)',K, iter));
    pause
    iter=iter+1;
end