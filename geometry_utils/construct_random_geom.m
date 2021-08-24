function randMat = construct_random_geom(TxN,nConds,nvp)
% Accepts as input a TxN matrix where T = C*M, with C being the number of
% conditions, M the number of trials per condition, and T the total number
% of trials. MxN is thus a point cloud around each empirical cluster
% centroid. Output is also a TxN matrix, however this TxN matrix consists
% of C random cluster centroids (sampled from a standard gaussian
% distribution, with the variance across these centroids rescaled to match
% the variance across the means of each of the C empirical points clouds,
% i.e. the empirical cluster centroids), each of which is surrounded by the
% same point cloud as that of the corresponding empirical cluster centroid,
% but rotated (by permuting neurons).
%
% PARAMETERS
% ----------
% TxN    -- nTrials x nNeurons matrix. The i_th row j_th column element is
%           the firing rate of the j_th neuron in the i_th trial. nTrials
%           can be evenly divisisble by the number of conditions, with each
%           condition represented by an even number of conditions.
% nConds -- Number of task conditions represented among the trials (first
%           dimension of TxN). 
% Name-Value Pairs (nvp)
%   'addNoise' -- This name value pair takes on a value of logical true or
%                 false and affects what is returned by the function. If
%                 true, the point cloud for each cluster is rotated about
%                 its centroid and moved to the location of the new
%                 randomly sampled cluster centroid; thus, the function
%                 returns a TxN matrix, with T = nTrials. If false, no
%                 point clouds are added to cluster centroids, and a CxN
%                 matrix is thus returned, with C = nConds.                
% 
% RETURNS
% -------
% randMat -- Either a TxN matrix (with T = nTrials and N = nNeurons) or a
%            CxN matrix (with C = nConds and N = nNeurons), depending on
%            the value of the 'addNoise' name value pair (see PARAMETERS
%            above). If TxN, the i_th row j_th column element is the firing
%            rate of the j_th neuron on the j_th single trial. If CxN, the
%            i_th row j_th column element is the firing rate of the i_th
%            neuron averaged over all trials of the j_th condition.
%
% Author: Jonathan Chien Version 1.0. 7/19/21. Last edit: 7/21/21.
% Based on methods section from Bernardi et al "The Geometry of Abstraction
% in the Hippocampus and Prefrontal Cortex," Cell, 2020.


arguments
    TxN
    nConds (1,1)
    nvp.addNoise = true 
end

% Get number of neurons and trials and determine number of trials per
% condition.
nNeurons = size(TxN, 2);
nTrials = size(TxN, 1);
nTrialsPerCond = nTrials / nConds;
assert(mod(nTrials,nConds)==0, 'nTrials must be evenly divisible by nConds.')

% Calculate mean of each condition (centroid of each empirical condition
% cluster).
meanCxN = calc_centroids(TxN, nConds);

% Calculate variance across empirical cluster centroids.
varCxN = var(meanCxN, 1);

% Sample C vectors from isotropic gaussian distribution and rescale by
% variance across empirical cluster centroids.
randomCentroids = randn(nConds, nNeurons) .* varCxN;

% Option to rotate empirical point clouds and move them to the new random
% cluster centroid locations.
if nvp.addNoise

    % For each condition, take mean centered point cloud (MxN) from
    % empirical data, rotate cloud by permuting neurons and add rotated,
    % mean-centered point cloud to synthetic centroid.
    nullTxN = NaN(size(TxN));
    for iCond = 1:nConds

        % Obtain point cloud around current condition.
        MxN = (TxN((iCond-1)*nTrialsPerCond + 1 : iCond*nTrialsPerCond, :));

        % Obtain mean centered point cloud.
        meanCenteredMxN = MxN - mean(MxN);

        % Perform rotation by permuting neurons.
        permMeanCenteredMxN = meanCenteredMxN(:,randperm(nNeurons));

        % Add rotated mean centered point cloud to current random cluster
        % centroid (this is essentially treating the random vector defining
        % the cluster centroid as a disaplacement vector applied to each
        % row/point of the mean-centered point cloud.
        randomCluster = randomCentroids(iCond,:) + permMeanCenteredMxN;

        % Assign MxN into correct section of function output.
        nullTxN((iCond-1)*nTrialsPerCond + 1 : iCond*nTrialsPerCond, :) ...
            = randomCluster;
    end
    
    % If noise was added, return the trials x neurons matrix (where each
    % trial is a point in the point clouds around the centroids).
    randMat = nullTxN;
else
    % If no noise added, return the cluster centroids (conditions x
    % neurons matrix).
    randMat = randomCentroids;
end

end