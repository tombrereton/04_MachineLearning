%% Preamble
clear all; close all; clc;
savePlots = 0;
pool = parpool;
stream = RandStream('mlfg6331_64');
options = statset('UseParallel',1,'UseSubstreams',1,'Streams',stream);

%% Load the data
load('cbt3data.mat');
X = diseased(:,:,1)';
Y = healthy(:,:,1)';

%% Initialise Km
Km_diseased = zeros(size(X, 1));
Km_healthy = zeros(size(Y, 1));
person_type = {'DiseasedPatients';'HealthyControls'};

%% We get a community matrix for 10,20, and 30 clusters
for K = 10:10:30;
    %% Get average community matrix for all (20) patients
    tic;
    for person = 1:size(diseased,3); % iterate over each person and take average

        % we get the ith diseased and healthy person
        X = diseased(:,:,person)'; 
        Y = healthy(:,:,person)'; % we get the ith healthy person

        % For the ith person, we get the cluster index for each brain region
        [ClusterIndex_diseased] = kmeans(X,K, 'Replicates',500, 'Options', options); 
        [ClusterIndex_healthy] = kmeans(Y,K, 'Replicates',500, 'Options', options); 

        % We compare the cluster index of the ith brain region with every
        % other brain region index, we return 1 if identical, 0 if not.
        % This produces a 90 by 90 matrix (90 brain regions)
        for i = 1:90;
            Km_diseased(:,i) = double(repmat(ClusterIndex_diseased(i,1),90,1) == ClusterIndex_diseased);
            Km_healthy(:,i) = double(repmat(ClusterIndex_healthy(i,1),90,1) == ClusterIndex_healthy);
        end

        % We initialise the mean community matrix
        if (person == 1);
            meanKm_diseased = Km_diseased;
            meanKm_healthy = Km_healthy;
        end

        % We calculate the running mean of the diseased community matrix
        Km_3d_diseased = cat(3,meanKm_diseased, Km_diseased);
        meanKm_diseased = mean(Km_3d_diseased,3);

        % We calculate the running mean of the healthy community matrix
        Km_3d_healthy = cat(3,meanKm_healthy, Km_healthy);
        meanKm_healthy = mean(Km_3d_healthy,3);        
    end

    %% View matrix as image (community matrix)
    meanKm_diseased_healthy = cat(3, meanKm_diseased, meanKm_healthy);

    for personType = 1:2
        figure(personType);
        imagesc(meanKm_diseased_healthy(:,:,personType));
        ti = sprintf('%s Community Matrix, for %g clusters',person_type{personType}, K);
        xlabel('Brain region','fontsize',16);
        ylabel('Brain region','fontsize',16);
        title(ti, 'fontsize',18);
        colorbar('eastoutside');

        if (savePlots == 1)
            name = sprintf('%s%g',person_type{personType},K);
            filename = sprintf(strcat(name,'.png')); 
            saveas(gcf,filename);
        end
        
        figure(3)
        imagesc(meanKm_healthy - meanKm_diseased);
        ti = sprintf('Difference Community Matrix, for %g clusters', K);
        xlabel('Brain region','fontsize',16);
        ylabel('Brain region','fontsize',16);
        title(ti, 'fontsize',18);
        colorbar('eastoutside');
        
        if (savePlots == 1)
            name = sprintf('diffCommunityMatrix%g',K);
            filename = sprintf(strcat(name,'.png')); 
            saveas(gcf,filename);
        end

    end
    toc;
end

%% Stops pool of workers
delete(gcp)