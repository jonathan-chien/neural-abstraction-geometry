function [ps,varargout] = calc_ps(TxN,trialLabels,nvp)
% Calculates "parallelsim score" (PS) as described in Bernardi et al "The
% Geometry of Abstraction in the Hippocampus and Prefrontal Cortex," Cell,
% 2020.
% 
% [ps] = calc_ps(TxN, trialLabels, [Name-Value Pairs])
% ---------------------------------------------------
% Returns struct containing the parallelism score for the given data, as
% well as associated metrics of statistical significance. Note that
% nConditions is calculated as the number of unique elements in the
% trialLabels vector.
%
% [ps, dichotomyConds] = calc_ps(TxN, trialLabels, [Name-Value Pairs])
% ---------------------------------------------------------------------
% Alternative syntax additionally returning an nDichotomies x nConds cell
% array containing the names of conditions associated with each dichotomy.
%
% PARAMETERS
% ----------
% TxN         -- T-by-N matrix, where T = nTrials and N = nNeurons. The
%                i_th row j_th column element is the firing rate of the
%                j_th neuron on the i_th single trial. All trials (rows of
%                TxN) belong to one of nConds conditions with nTrials >> 
%                nConds. Note that each of the nConds conditions is
%                represented nTrials / nConds times among the rows of TxN.
% trialLabels -- T-vector, with T = nTrials, where each element is a
%                positive integer condition label corresponding to a row
%                (single trial) of TxN.
% Name-Value Pairs (nvp)
%   'condLabels' -- 1 x nConds cell array, where each cell contains the
%                   name/description of a condition. Defualt is an empty
%                   array, in which case the optional argument
%                   dichotomyConds (see RETURNS below) will be returned as
%                   an empty array.
%   'dropIdc'    -- 1 x n vector, where each element contains the index of
%                   neuron within the population that we wish to drop (i.e.
%                   all its entries across trials are deleted, so that the
%                   length of the second dimension of TxN overall decreases
%                   by n) before calculating PS. 
%   'nNull'      -- Scalar integer that is the number of synthetic/null
%                   datasets to generate in order to calculate statistical
%                   significance of the parallelism score. Default = 1000.
%   'confInt'    -- Scalar integer specifying the confidence interval size
%                   (as a percentage) to be calculated around the mean of
%                   the null distributions. Default = 95.
%   'nullMethod' -- String value indicating the method to be used to
%                   generate a null dataset, either 'permutation' or
%                   'geometric'. If 'permutation', the original TxN input
%                   is shuffled for each neuron independently (i.e.,
%                   entries of each column vector are shuffled
%                   independently), then the process of calculating the PS
%                   is carried out as on the empirical data. This destroys
%                   cluster structure to a certain extent (though it will
%                   completely fail to do so in certain cases) and is
%                   equivalent to shuffling each neuron's trial labels
%                   independently (while preserving the marginal
%                   distribution of each neuron's firing rate across
%                   trials). If 'geometric', nConds N-vectors are sampled
%                   from a standard normal distribution and considered
%                   random cluster centroids, and point clouds
%                   corresponding to each condition in the empirical data
%                   are rotated and the moved to the new centroids in
%                   N-space; see construct_random_geom for more details on
%                   this process.
%   'pVal'       -- May take on values: 'two-tailed', 'left-tailed',
%                   'right-tailed', or logical false. If false, null
%                   distribution for PS will not be computed. If one of the
%                   above string values, null distribution will be computed
%                   for PS (see 'nullMethod' below), and p
%                   values/confidence intervals will be calculated
%                   accordingly. Default is 'two-tailed'.
% 
% RETURNS
% -------
% ps             -- 1 x 1 struct with the following fields:
%   .ps       -- nDichotomies x 1 vector, where the i_th element is the
%                parallelism score for the i_th dichotomy in the input
%                data.
%   .p        -- nDichotomies x 1 vector, where the i_th element is the p
%                value attached to the parallelism score for the i_th
%                dichotomy.
%   .confInt  -- nDichotomies x 2 matrix, where each row corresponds to a
%                dichotomy. For each row/dichotomy, the first and second
%                column elements are the upper and lower bounds,
%                respectively, of the confidence interval around the null
%                distribution mean; the size of this interval was specified
%                through the 'confInt' name value pair (default size is
%                95).
%   .obsStdev -- nDichotomies x 1 vector, where each element is the number
%                of standard deviations from the mean of the i_th
%                dichotomy's null distribution that lies the observed
%                parallelism score on the i_th dichotomy.
% dichotomyConds -- An optional returned value, dichtomyConds is an
%                   nDichotomies x nConds cell array where each row
%                   corresponds to a dichotomy. For each row (dichotomy),
%                   the first 1:nConds/2 cells contain the labels of the
%                   conditions on one side of the dichotomy and the last
%                   nConds/2+1:end cells contain the labels of the
%                   conditions on the other side of the dichotomy. If
%                   condLabels is empty (as it is by default), this value,
%                   if requested, will be returned as an empty array.
%
% Author: Jonathan Chien Version 1.1. 7/23/21. Last edit: 8/23/21.


arguments
    TxN
    trialLabels
    nvp.condLabels = []
    nvp.dropIdc = []
    nvp.nNull = 1000
    nvp.confInt = 95
    nvp.nullMethod = 'geometric'
    nvp.pVal = 'two-tailed'
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

% Get dichotomy indices and labels.
[dichotomies,dichotomyConds] = create_dichotomies(nConds, nvp.condLabels);
nDichotomies = size(dichotomies, 1);
varargout{1} = dichotomyConds;
if nargout == 2 && isempty(nvp.condLabels)
    warning(['Dichotomy labels requested but no condition labels were ' ...
             'supplied. dichotomyConds will be returned as an empty array.'])  
end

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
    
    % Obtain all permutations of side2. Also prepare indices of mchoose2
    % used to index pairs of coding vectors, calculate number of pairs, and
    % preallocate.
    side2Perms = perms(side2);
    nPerms = size(side2Perms, 1);
    pairIdc = nchoosek(1:m, 2); % each row is a pair of coding vecs
    nPairs = size(pairIdc, 1);
    codingVecPairs = NaN(2, nPairs, nNeurons);
    meanCosineSim = NaN(nPerms, 1);
    
    % Define coding vectors (one to one match) between side 1 and each
    % permutation of side 2; these are vector differences. E.g., there are
    % four coding vectors in the case of 8 conditions.
    for iPerm = 1:nPerms
        
        % Calculate vector difference between the i_th condition vector
        % (averaged over all trials of that condition) in side1 and the
        % i_th condition vector (also averaged over trials) in the current
        % permutation of side2 for all i from 1 to m.
        codingVecs = CxN(side1,:) - CxN(side2Perms(:,iPerm),:);
        
        % Determine all unique ways to pair up the coding vectors for the
        % current permutation, then calculate cosine similarity of all
        % pairs.
        codingVecPairs(1,:,:) = codingVecs(pairIdc(:,1),:);
        codingVecPairs(2,:,:) = codingVecs(pairIdc(:,2),:);
        normedCodingVecPairs = codingVecPairs ./ vecnorm(codingVecPairs, 2, 3);
        cosineSims = sum(normedCodingVecPairs(1,:,:) ...
                         .* normedCodingVecPairs(2,:,:), ...
                         3);
        meanCosineSim(iPerm) = mean(cosineSims);
    end
    
    % Take max of the mean cosine similarities from each permutation as the
    % PS for the current dichotomy.
    parScore(iDichot) = max(meanCosineSim);
end

ps.ps = parScore;


%% Compute null distribution for PS

if nvp.pVal
    
% Preallocate.
nullParScore = NaN(nDichotomies, nvp.nNull);

% Generate distribution of PS scores for each dichotomy by repeating above
% process 'nNull' times.
for iNull = 1:nvp.nNull    
    
    % Construct null model.
    switch nvp.nullMethod
        case 'permutation'
            nullTxN = NaN(size(TxN));
            nTrials = size(TxN, 1);
            
            for iNeuron = 1:nNeurons
                nullTxN(:,iNeuron) = TxN(randperm(nTrials),iNeuron);
            end 
            
            nullCxN = calc_centroids(nullTxN, nConds);
            
        case 'geometric'
            nullCxN = construct_random_geom(TxN, nConds, 'addNoise', false);
    end
    
    % Calculate parallelism score (PS) for each dichotomy.
    currParScore = NaN(nDichotomies, 1);
    for iDichot = 1:nDichotomies
    
        % Obtain indices of condtitions on either side of current dichotomy.
        side1 = dichotomies(iDichot,1:m);
        side2 = dichotomies(iDichot,m+1:end);

        % Obtain all permutations of side2. Also prepare indices of
        % mchoose2 used to index pairs of coding vectors, calculate number
        % of pairs, and preallocate.
        side2Perms = perms(side2);
        nPerms = size(side2Perms, 1);
        pairIdc = nchoosek(1:m, 2); % each row is a pair of coding vecs
        nPairs = size(pairIdc, 1);
        codingVecPairs = NaN(2, nPairs, nNeurons);
        meanCosineSim = NaN(nPerms, 1);

        % Define four coding vectors (one to one match) between side 1 and
        % each permutation of side 2. These are vector differences.
        for iPerm = 1:nPerms

            % Calculate all vector difference between the i_th condition
            % vector (averaged over all trials of that condition) in side1
            % and the i_th condition vector (also averaged over trials) in
            % the current permutation of side2.
            codingVecs = nullCxN(side1,:) - nullCxN(side2Perms(:,iPerm),:);

            % Determine all unique ways to pair up the coding vectors for
            % the current permutation, then calculate cosine similarity of
            % all pairs.
            codingVecPairs(1,:,:) = codingVecs(pairIdc(:,1),:);
            codingVecPairs(2,:,:) = codingVecs(pairIdc(:,2),:);
            normedCodingVecPairs = codingVecPairs ./ vecnorm(codingVecPairs, 2, 3);
            cosineSims = sum(normedCodingVecPairs(1,:,:) ...
                             .* normedCodingVecPairs(2,:,:), ...
                             3);
            meanCosineSim(iPerm) = mean(cosineSims);
        end

        % Take max of the mean cosine similarities from each permutation as
        % the PS for the current dichotomy.
        currParScore(iDichot) = max(meanCosineSim);
    end
    
    % Store PS of all dichotomies from current null run.
    nullParScore(:,iNull) = currParScore;
end


%% Compute p values and confidence intervals

% Calculate p value.
ps.p = tail_prob(parScore, nullParScore, 'type', nvp.pVal);

% Caclulate confidence interval around mean of null distribution. 
ps.confInt = montecarlo_conf_int(nullParScore, nvp.confInt);

% Calculate number of st.dev.'s away from null mean that empirical value
% lies.
ps.obsStdev = get_num_stdev(parScore, nullParScore);

end

end

% function CxN = calculate_centroids(TxN, nConds)
% % Calculate condition cluster centroids by averaging across all trials
% % within each condition. Input is nTrials x nNeurons matrix of firing
% % rates, returns nConds x nNeurons of firing rates (corresponding to
% % cluster centroids).
% 
% % Determine number of neurons, trials, and trials per condition.
% [nTrials, nNeurons] = size(TxN);
% nTrialsPerCond = nTrials / nConds;
% assert(mod(nTrials,nConds)==0, 'nTrials must be evenly divisible by nConds.')
% 
% % Average within condition for all conditions.
% CxN = NaN(nConds, nNeurons);
% for iCond = 1:nConds
%     CxN(iCond,:) ...
%         = mean(TxN((iCond-1)*nTrialsPerCond + 1 : iCond*nTrialsPerCond, :));
% end
% 
% end
