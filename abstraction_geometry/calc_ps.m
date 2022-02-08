function ps = calc_ps(TxN,trialLabels,nvp)
% Calculates "parallelsim score" (PS) as described in Bernardi et al "The
% Geometry of Abstraction in the Hippocampus and Prefrontal Cortex," Cell,
% 2020. Returns struct containing the parallelism score for the given data,
% as well as associated metrics of statistical significance.
%
% PARAMETERS
% ----------
% TxN         -- T-by-N matrix, where T = nTrials and N = nNeurons. The
%                i_th row j_th column element is the firing rate of the
%                j_th neuron on the i_th single trial. All trials (rows of
%                TxN) belong to one of nConds conditions with nTrials >> 
%                nConds. Note that each of the nConds conditions is
%                represented nTrials / nConds times among the rows of TxN.
% trialLabels -- Vector of length = nTrials, where each element is a
%                positive integer condition label corresponding to a row
%                (single trial) of TxN. The number of conditions is
%                calculated as the number of unique elements in the
%                trialLabels vector.
% Name-Value Pairs (nvp)
%   'condLabels'     -- 1 x nConds cell array, where each cell contains the
%                       name/description of a condition. Default is an
%                       empty array, in which case the ps struct will be
%                       returned without the dichotomyConds field (see
%                       RETURNS).
%   'dropIdc'        -- 1 x n vector, where each element contains the index
%                       of neuron within the population that we wish to
%                       drop (i.e. all its entries across trials are
%                       deleted, so that the length of the second dimension
%                       of TxN overall decreases by n) before calculating
%                       PS.
%   'nNull'          -- Scalar integer that is the number of synthetic/null
%                       datasets to generate in order to calculate
%                       statistical significance of the parallelism score.
%                       Default = 1000.
%   'nullInt'        -- Scalar integer specifying the interval size (as a
%                       percentage) to be calculated around the mean of the
%                       null distributions. Default = 95.
%   'nullMethod'     -- String value indicating the method to be used to
%                       generate a null dataset, either 'permutation' or
%                       'geometric'. If 'permutation', the original TxN input
%                       is shuffled for each neuron independently (i.e.,
%                       entries of each column vector are shuffled
%                       independently), then the process of calculating the PS
%                       is carried out as on the empirical data. This destroys
%                       cluster structure to a certain extent (though it will
%                       completely fail to do so in certain cases) and is
%                       equivalent to shuffling each neuron's trial labels
%                       independently (while preserving the marginal
%                       distribution of each neuron's firing rate across
%                       trials). If 'geometric', nConds N-vectors are sampled
%                       from a standard normal distribution and considered
%                       random cluster centroids, and point clouds
%                       corresponding to each condition in the empirical data
%                       are rotated and the moved to the new centroids in
%                       N-space; see construct_random_geom for more details on
%                       this process.
%   'pVal'           -- May take on values: 'two-tailed', 'left-tailed',
%                       'right-tailed', or logical false. If false, null
%                       distribution for PS will not be computed. If one of
%                       the above string values, null distribution will be
%                       computed for PS (see 'nullMethod' below), and p
%                       values/null intervals will be calculated
%                       accordingly. Default is 'two-tailed'.
%   'returnNullDist' -- (1|0, default = 0). Specify whether or not to
%                       return the null distribution for the parallelism
%                       score (for each dichotomy) in the ps struct (see
%                       RETURNS below).
% 
% RETURNS
% -------
% ps             -- 1 x 1 struct with the following fields:
%   .ps             -- nDichotomies x 1 vector, where the i_th element is
%                      the parallelism score for the i_th dichotomy in the
%                      input data.
%   .p              -- nDichotomies x 1 vector, where the i_th element is
%                      the p value attached to the parallelism score for
%                      the i_th dichotomy.
%   .nullInt        -- nDichotomies x 2 matrix, where each row corresponds
%                      to a dichotomy. For each row/dichotomy, the first
%                      and second column elements are the upper and lower
%                      bounds, respectively, of the specified interval
%                      around the null distribution mean; the size of this
%                      interval was specified through the 'nullInt' name
%                      value pair (default size is 95).
%   .obsStdev       -- nDichotomies x 1 vector, where each element is the
%                      number of standard deviations from the mean of the
%                      i_th dichotomy's null distribution that lies the
%                      observed parallelism score on the i_th dichotomy.
%   .nullDist       -- nDichotomies x nNull array (see 'nNull' under
%                      Name-Value Pair options above) whose i_th j_th
%                      element is the j_th draw from the parallelism score
%                      null distribution for the i_th dichotomy.
%   .dichotomyConds -- An optionally returned field, dichtomyConds is an
%                      nDichotomies x nConds cell array where each row
%                      corresponds to a dichotomy. For each row
%                      (dichotomy), the first 1:nConds/2 cells contain the
%                      labels of the conditions on one side of the
%                      dichotomy and the last nConds/2+1:end cells contain
%                      the labels of the conditions on the other side of
%                      the dichotomy. If condLabels is empty (as it is by
%                      default), this field will be absent from ps.
%
% Author: Jonathan Chien 7/23/21. Last edit: 2/4/22.

arguments
    TxN
    trialLabels
    nvp.condLabels = []
    nvp.dropIdc = []
    nvp.nNull = 1000
    nvp.nullInt = 95
    nvp.nullMethod = 'geometric'
    nvp.pVal = 'two-tailed'
    nvp.returnNullDist = false
end


%% Preprocess inputs

% Option to drop neurons if desired, then obtain number of neurons.
TxN(:,nvp.dropIdc) = [];
nNeurons = size(TxN, 2);

% Determine number of conditions and set combination parameter m, where we
% want to partition m*n objects into n unique groups, with n = 2 (hence a
% "dichotomy").
nConds = length(unique(trialLabels));
m = nConds / 2;

% Get dichotomy indices and labels (if provided). Store labels as field of
% ps at end to preserve order of fields.
[dichotomies,dichotomyConds] = create_dichotomies(nConds, nvp.condLabels);
nDichotomies = size(dichotomies, 1);

% Calculate mean of each condition (centroid of each empirical condition
% cluster).
CxN = calc_centroids(TxN, nConds);


%% Calculate Parallelism Score (PS)

% Preallocate.
parScore = NaN(nDichotomies, 1);

% Calculate parallelism score (PS) for all dichotomies.
parfor iDichot = 1:nDichotomies
    
    % Obtain indices of condtitions on either side of current dichotomy.
    side1 = dichotomies(iDichot,1:m);
    side2 = dichotomies(iDichot,m+1:end);
    
    % Take max of the mean cosine similarities from each permutation as the
    % PS for the current dichotomy.
    parScore(iDichot) = max(linking_vecs(CxN, side1, side2));
end

ps.ps = parScore;


%% Compute null distribution for PS

if nvp.pVal
    
% Preallocate.
nullParScore = NaN(nDichotomies, nvp.nNull);

% Generate distribution of PS scores for each dichotomy by repeating above
% process 'nNull' times.
parfor iNull = 1:nvp.nNull    
    
    % Construct null model via permutation.
    if strcmp(nvp.nullMethod, 'permutation')
        nullTxN = NaN(size(TxN));
        nTrials = size(TxN, 1);
        
        for iNeuron = 1:nNeurons
            nullTxN(:,iNeuron) = TxN(randperm(nTrials),iNeuron);
        end 
        
        nullCxN = calc_centroids(nullTxN, nConds);
       
    % Construct null model via geometric model.
    elseif strcmp(nvp.nullMethod, 'geometric')
        nullCxN = construct_random_geom(TxN, nConds, 'addNoise', false);
    end
    
    % Calculate parallelism score (PS) for each dichotomy.
    currParScore = NaN(nDichotomies, 1);
    for iDichot = 1:nDichotomies
    
        % Obtain indices of condtitions on either side of current dichotomy.
        side1 = dichotomies(iDichot,1:m);
        side2 = dichotomies(iDichot,m+1:end);

        % Take max of the mean cosine similarities from each permutation as
        % the PS for the current dichotomy.
        currParScore(iDichot) = max(linking_vecs(nullCxN, side1, side2));
    end
    
    % Store PS of all dichotomies from current null run.
    nullParScore(:,iNull) = currParScore;
end

% Option to return null distribution of PS for each dichotomy.
if nvp.returnNullDist, ps.nullDist = nullParScore; end


%% Compute p values and null intervals

% Calculate p value.
ps.p = tail_prob(parScore, nullParScore, 'type', nvp.pVal);

% Caclulate interval around mean of null distribution. 
ps.nullInt = montecarlo_int(nullParScore, nvp.nullInt);

% Calculate number of st.dev.'s away from null mean that empirical value
% lies.
ps.obsStdev = get_num_stdev(parScore, nullParScore);

end

% Add names of conditions in dichotomies if condition labels provided.
if ~isempty(dichotomyConds), ps.dichotomyConds = dichotomyConds; end

end


% --------------------------------------------------
function meanCosineSim = linking_vecs(CxN, side1, side2)
% For a given dichotomy (each side's conditions indices are in side1 and
% side2), compute the vector differences (linking or coding vectors)
% between condition centroids from side1 and side2 across all possible ways
% of matching condition centroids from the two sides of the dichotomy.
%
% PARAMETERS
% ----------
% CxN   -- nConditions x nNeurons matrix of firing rates.
% side1 -- Vector of length nConditions/2 containing condition indices from
%          side 1 of the current dichotomy.
% side2 -- Vector of length nConditions/2 containing condition indices from
%          side 2 of the current dichotomy.
%
% RETURNS
% -------
% meanCosineSim - nPermutations x 1 vector whose i_th element is the
%                 mean cosine similarity across all pairs of linking vectors
%                 for one way (permutation of side2) of matching up
%                 condition centroids from the two sides of the dichotomy.
%
% Jonathan Chien. 8/23/21.

% Number of neurons and conditions.
[nConds, nNeurons] = size(CxN); 
m = nConds/2;

% Obtain all permutations of side2. Also prepare indices of mchoose2
% used to index pairs of linking vectors, calculate number of pairs, and
% preallocate.
side2Perms = perms(side2);
nPerms = size(side2Perms, 1);
pairIdc = nchoosek(1:m, 2); % each row is a pair of linking vecs
nPairs = size(pairIdc, 1);
linkingVecPairs = NaN(2, nPairs, nNeurons);
meanCosineSim = NaN(nPerms, 1);

% Define linking vectors (one to one match) between side 1 and each
% permutation of side 2; these are vector differences. E.g., there are
% four linking vectors in the case of 8 conditions.
for iPerm = 1:nPerms
    
    % Calculate vector difference between the i_th condition vector
    % (averaged over all trials of that condition) in side1 and the
    % i_th condition vector (also averaged over trials) in the current
    % permutation of side2 for all i from 1 to m.
    linkingVecs = CxN(side1,:) - CxN(side2Perms(iPerm,:),:);
    
    % Determine all unique ways to pair up the linking vectors for the
    % current permutation, then calculate cosine similarity of all
    % pairs.
    linkingVecPairs(1,:,:) = linkingVecs(pairIdc(:,1),:);
    linkingVecPairs(2,:,:) = linkingVecs(pairIdc(:,2),:);
    normedlinkingVecPairs = linkingVecPairs ./ vecnorm(linkingVecPairs, 2, 3);
    cosineSims = sum(normedlinkingVecPairs(1,:,:) ...
                     .* normedlinkingVecPairs(2,:,:), ...
                     3);
    meanCosineSim(iPerm) = mean(cosineSims);
end

end