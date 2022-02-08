function ccgp = calc_ccgp(TxN,trialLabels,nvp)
% Implementation of Cross-Condition Generalization Performance (CCGP) from
% Bernardi et al "The Geometry of Abstraction in the Hippocampus and
% Prefrontal Cortex," Cell, 2020. For all possible ways of dividing the
% nConds conditions in two (with each unique division termed a dichotomy),
% returns struct containing "cross-condition generalization performance"
% (CCGP) of each dichotomy, as well as measures of the statistical
% signficance of each dichotomy.
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
%                       empty array, in which case the ccgp struct will be
%                       returned without the dichotomyConds field (see
%                       RETURNS). Note that the returned ccgp struct will
%                       also lack a dichotomyConds field if a custom matrix
%                       of dichotomy indices is passed in through the
%                       'dichotomies' name-value pair, even if 'condLabels'
%                       is passed in with a valid nonempty value.
%   'dropIdc'        -- 1 x n vector, where each element contains the index
%                       of neuron within the population that we wish to
%                       drop (i.e. all its entries across trials are
%                       deleted, so that the length of the second dimension
%                       of TxN overall decreases by n) before calculating
%                       CCGP.
%   'dichotomies'    -- nDichotomies x nCond matrix where the first and
%                       halves of the elements in the i_th row are the
%                       indicies of the conditions on each side of the i_th
%                       dichotomy. Same format as the output of
%                       create_dichotomies.m, but here the user can specify
%                       a custom matrix to only test a subset of
%                       dichotomies (which saves time if the CCGP of only a
%                       few dichotomies is desired). By default, this value
%                       is an empty array, in which case this function
%                       automatically generates and tests all possible
%                       dichotomies for the supplied data.
%   'classifier'     -- Function handle for classifier serving as decoder.
%                       Default is @fitclinear, which fits a linear SVM
%                       model by default.
%   'pVal'           -- May take on values: 'two-tailed', 'left-tailed',
%                       'right-tailed', or logical false. If false, null
%                       distributions for CCGP (accuracy and AUC) will not
%                       be computed. If one of the above string values,
%                       null distributions will be computed for CCGP based
%                       on both accuracy and AUC (see 'nullMethod' below),
%                       and p values will be calculated accordingly.
%                       Default is 'two-tailed'.
%   'nullInt'        -- Scalar integer specifying the interval size (as a
%                       percentage) to be calculated around the mean of the
%                       null distributions. Default = 95.
%   'nNull'          -- Scalar integer that is the number of synthetic/null
%                       datasets to generate in order to calculate
%                       statistical significance of CCGP.
%   'nullMethod'     -- String value indicating the method to be used to
%                       generate a null dataset, either 'permutation' or
%                       'geometric'. If 'permutation', the original TxN
%                       input is shuffled for each neuron independently
%                       (i.e., entries of each column vector are shuffled
%                       independently), then the process of calculating
%                       CCGP is carried out as on the empirical data. This
%                       destroys cluster structure to a certain extent
%                       (though it will completely fail to do so in certain
%                       cases) and is equivalent to shuffling each neuron's
%                       trial labels independently (while preserving the
%                       marginal distribution of each neuron's firing rate
%                       across trials). If 'geometric', nConds N-vectors
%                       are sampled from a standard normal distribution and
%                       considered random cluster centroids, and point
%                       clouds corresponding to each condition in the
%                       empirical data are rotated and the moved to the new
%                       centroids in N-space; see construct_random_geom for
%                       more details on this process.
%   'permute'        -- String value, either 'trialLabels' or 'neurons'. If
%                       'trialLabels', the permutation null model (see
%                       'nullMethod' above) is generated by permuting the
%                       elements of the trialLabels vector but leaving the
%                       firing rate matrix, TxN, intact. If 'neurons', then
%                       each the permutation null is generated via
%                       independently permuting each of the columns of the
%                       firing rate data matrix, TxN. Note that this
%                       argument is ignored if 'nullMethod' = 'geometric'.
%                       Default = 'neurons'.
%   'returnNullDist' -- (1|0, default = 0). Specify whether or not to
%                       return the null distributions for accuracy and auc
%                       (for each dichotomy) in the ccgp struct (see
%                       RETURNS below).
%
% RETURNS
% -------
% ccgp           -- 1 x 1 struct with the following fields:
%   .accuracy       -- 1 x 1 struct with the following fields containing
%                      CCGP information based on accuracy metric.
%       .ccgp     -- nDichotomies x 1 vector whose elements are the decoder
%                    accuracy for each dichotomy, averaged over all 
%                    possible decoders (i.e., all ways of choosing training
%                    and testing conditions) for that dichotomy.
%       .p        -- nDichotomies x 1 vector whose elements are p values
%                    for the CCGP of each dichotomy as measured by decoder
%                    accuracy.
%       .nullInt  -- nDichotomies x 2 matrix, where each row corresponds to
%                    one dichotomy. For each row/dichotomy, the first and
%                    second column elements are the upper and lower bounds,
%                    respectively, of the specified interval around the
%                    null distribution mean; the size of this interval was
%                    specified through the 'nullInt' name value pair
%                    (default size is 95).
%       .obsStdev -- nDichotomies x 1 vector, where each element is the
%                    number of standard deviations from the mean of the
%                    null CCGP (accuracy) distribution that the empirical
%                    CCGP for that dichotomy lies.
%       .nullDist -- nDichotomies x nNull array (see 'nNull' under
%                    Name-Value Pair options above) whose i_th j_th element
%                    is the j_th draw from the null accuracy-based CCGP
%                    distribution for the i_th dichotomy. 
%   .auc            -- 1 x 1 struct with the following fields containing
%                      CCGP information based on AUC metric.
%       .ccgp     -- nDichotomies x 1 vector whose elements are the decoder
%                    AUC for each dichotomy, averaged over all possible
%                    decoders (i.e., all ways of choosing training and
%                    testing conditions) for that dichotomy.
%       .p        -- nDichotomies x 1 vector whose elements are p values 
%                    for the CCGP of each dichotomy as measured by decoder
%                    AUC.
%       .nullInt  -- nDichotomies x 2 matrix, where each row corresponds to
%                    one dichotomy. For each row/dichotomy, the first and
%                    second column elements are the upper and lower bounds,
%                    respectively, of the specified interval around the
%                    null distribution mean; the size of this interval was
%                    specified through the 'nullInt' name value pair
%                    (default size is 95).
%       .obsStdev -- nDichotomies x 1 vector, where each element is the
%                    number of standard deviations from the mean of the
%                    null CCGP (AUC) distribution that the empirical CCGP
%                    for that dichotomy lies.
%       .nullDist -- nDichotomies x nNull array (see 'nNull' under
%                    Name-Value Pair options above) whose i_th j_th element
%                    is the j_th draw from the null AUC-based CCGP
%                    distribution for the i_th dichotomy. 
%   .dichotomyConds -- An optionally returned field, dichtomyConds is an
%                      nDichotomies x nConds cell array where each row
%                      corresponds to a dichotomy. For each row
%                      (dichotomy), the first 1:nConds/2 cells contain the
%                      labels of the conditions on one side of the
%                      dichotomy and the last nConds/2+1:end cells contain
%                      the labels of the conditions on the other side of
%                      the dichotomy. If condLabels is empty (as it is by
%                      default), this field will be absent from ccgp. Note
%                      as well that this field will be absent from ccgp if
%                      a custom set of dichotomies is passed in through
%                      'dichotomies' (see PARAMETERS), even if 'condLabels'
%                      is passed with a valid nonempty value.
%
% Author: Jonathan Chien 7/16/21. Last edit: 2/4/22.

arguments
    TxN
    trialLabels
    nvp.condLabels = []
    nvp.dropIdc = []
    nvp.dichotomies {mustBeNumeric} = []
    nvp.classifier = @fitclinear
    nvp.pVal = 'two-tailed'
    nvp.nullInt (1,1) = 95
    nvp.nNull (1,1) = 2
    nvp.nullMethod string = 'geometric'
    nvp.permute string = 'neurons'
    nvp.returnNullDist = false
end


%% Set parameters based on inputs and prepare for decoding runs

% Option to drop neurons if desired.
TxN(:,nvp.dropIdc) = [];

% Determine number of conditions.
nConds = length(unique(trialLabels));

% Set combination parameter m, where we want to partition m*n objects into
% n unique groups, with n = 2 (hence a "dichotomy").
m = nConds/2;

% Create struct to presever order of fields if dichotomyConds is added.
ccgp = struct('accuracy', cell(1, 1), 'auc', cell(1, 1));
   
% If user did not pass in dichotomies, generate all possible dichotomies
% and store corresponding condition labels in cell array dichotomyConds.
if isempty(nvp.dichotomies)
    [dichotomies, dichotomyConds] = create_dichotomies(nConds, nvp.condLabels);
    nDichotomies = size(dichotomies, 1);
    if ~isempty(dichotomyConds), ccgp.dichotomyConds = dichotomyConds; end
else
    dichotomies = nvp.dichotomies;
    nDichotomies = size(dichotomies, 1);
end

% Calculate number of unique decoders trained for each dichotomy so that we
% can preallocate.
nDecoders = 0;
for k = 1:m-1
    nDecoders = nDecoders + (nchoosek(m,k))^2;
end
dichotDecodersAcc = NaN(nDichotomies, nDecoders);
dichotDecodersAuc = NaN(nDichotomies, nDecoders);


%% Calculate CCGP for all dichotomies in empirical data

% Calculate CCGP for each dichotomy.
parfor iDichot = 1:nDichotomies
                    
    % Get condition labels corresponding to either side of the current
    % dichotomy.
    side1 = dichotomies(iDichot, 1:m);
    side2 = dichotomies(iDichot, m+1:end);
    
    % Calculate and store performances of all decoders for current
    % dichotomy.
    [dichotDecodersAcc(iDichot,:), dichotDecodersAuc(iDichot,:)] ...
        = test_dichotomy(TxN, trialLabels, side1, side2, nvp.classifier);
end

% Store results, including both the average across all decoders for each
% dichotomy (CCGP) and the full distribution of all decoders' performances
% for each dichotomy.
ccgp.accuracy.ccgp = mean(dichotDecodersAcc, 2); 
ccgp.auc.ccgp = mean(dichotDecodersAuc, 2);
ccgp.accuracy.dichotDecoders = dichotDecodersAcc; 
ccgp.auc.dichotDecoder = dichotDecodersAuc;


%% Generate null distribution for CCGP

if nvp.pVal

% Determine number of trials and neurons. Preallocate.
nTrials = size(TxN, 1); 
nNeurons = size(TxN, 2);
nullDichotAcc = NaN(nDichotomies, nvp.nNull);
nullDichotAuc = NaN(nDichotomies, nvp.nNull);

% Repeatedly test CCGP over null datasets.
parfor iNull = 1:nvp.nNull
    
    % Generate null model via permutation.
    if strcmp(nvp.nullMethod, 'permutation')
        % Permute trial labels but leave data (TxN) intact.
        if strcmp(nvp.permute, 'trialLabels')            
            nullTrialLabels = trialLabels(randperm(nTrials));
            nullTxN = TxN;

        % Independently permute each column (neuron) of TxN but leave 
        % trialLabels intact.
        elseif strcmp(nvp.permute, 'neurons')
            nullTrialLabels = trialLabels;
            nullTxN = NaN(size(TxN));
            for iNeuron = 1:nNeurons
                nullTxN(:,iNeuron) = TxN(randperm(nTrials),iNeuron);
            end
        end
     
    % Generate null model via geometric model. Retain trial labels.
    elseif strcmp(nvp.nullMethod, 'geometric')        
        nullTrialLabels = trialLabels;
        nullTxN = construct_random_geom(TxN, nConds, 'addNoise', true); 
        
    else
        error("Invalid value for 'nullMethod'.")
    end
    
    % Preallocate matrices to hold all values across all dichotomies and
    % decoders for current null run (this helps with indexing/slicing using
    % the parfor loop).
    currNullDichotDecoderAcc = NaN(nDichotomies, nDecoders);
    currNullDichotDecoderAuc = NaN(nDichotomies, nDecoders);
    
    % For current null dataset, iterate over all dichotomies and decoders.
    for iDichot = 1:nDichotomies
        
        % Get condition labels corresponding to either side of the current
        % dichotomy.
        side1 = dichotomies(iDichot, 1:m);
        side2 = dichotomies(iDichot, m+1:end);
        
        % Test current dichotomy's CCGP on null data/labels.
        [currNullDichotDecoderAcc(iDichot,:), ...
         currNullDichotDecoderAuc(iDichot,:)] ...
            = test_dichotomy(nullTxN, nullTrialLabels, side1, side2, nvp.classifier);
    end
    
    % Average over decoders and store results from current null run. We
    % will not store individual decoder performances within dichotomies.
    nullDichotAcc(:,iNull) = mean(currNullDichotDecoderAcc, 2);
    nullDichotAuc(:,iNull) = mean(currNullDichotDecoderAuc, 2);
end

% Option to return null distribution (for each dichotomy individually).
if nvp.returnNullDist
    ccgp.accuracy.nullDist = nullDichotAcc;
    ccgp.auc.nullDist = nullDichotAuc;
end


%% Calculate p values and st.dev. under null

% Calculate p values.
ccgp.accuracy.p = tail_prob(ccgp.accuracy.ccgp, nullDichotAcc, 'type', nvp.pVal);
ccgp.auc.p = tail_prob(ccgp.auc.ccgp, nullDichotAuc, 'type', nvp.pVal);

% Get values at 2 st.dev. in null distribution. 
ccgp.accuracy.nullInt = montecarlo_int(nullDichotAcc, nvp.nullInt);
ccgp.auc.nullInt = montecarlo_int(nullDichotAuc, nvp.nullInt);

% Calculate number of st.dev.'s away from mean that empirical value lies.
ccgp.accuracy.obsStdev = get_num_stdev(ccgp.accuracy.ccgp, nullDichotAcc);
ccgp.auc.obsStdev = get_num_stdev(ccgp.auc.ccgp, nullDichotAuc);

end

end


% --------------------------------------------------
function [accuracy,auc] = test_dichotomy(TxN,trialLabels,side1,side2,classifier)
% For a single dichotomy, this function tests all possible ways of choosing 
% training and testing subsets on each side of the dichotomy, as a test of
% representational abstraction.
%
% PARAMETERS
% ----------
% TxN         -- nTrials x nNeurons array of firing rates.
% trialLabels -- nTrials x 1 vector of labels identifying the task
%                condition of the respective single trials in the rows of
%                TxN.
% side1       -- Vector of length nConditions/2 with the indices of
%                conditions on one side of the dichotomy.
% side2       -- Vector of length nConditions/2 with the indices of
%                conditions on the other side of the dichotomy (with
%                respect to side1).
% classifier  -- Function handle for desired classifier serving as decoder.
%
% RETURNS
% -------
% accuracy -- nDecoders x 1 array of accuracy values (one for each of the
%             decoders arising from a unique way of choosing training and
%             testing subsets on both sides of the dichotomy).
% auc      -- nDecoders x 1 array of AUC values (one for each of the
%             decoders arising from a unique way of choosing training and
%             testing subsets on both sides of the dichotomy).
%
% Author: Jonathan Chien. 1/20/22.


% Determine number of conditions.
nConds = length(unique(trialLabels));

% Set combination parameter m, where we want to partition m*n objects into
% n unique groups, with n = 2 (hence a "dichotomy").
m = nConds/2;

% Calculate number of unique decoders trained for each dichotomy so that we
% can preallocate.
nDecoders = 0;
for k = 1:m-1
    nDecoders = nDecoders + (nchoosek(m,k))^2;
end

% Preallocate and initialize index to track decoders across all possible
% choices of train and test sets.
accuracy = NaN(nDecoders, 1);
auc = NaN(nDecoders, 1);
iDecoder = 0;

% Train over all possible combinations of conditions. First, vary the
% number of conditions, k, subsampled for training.
for k = 1:m-1
    
    % We would like to sample k condition labels from each side to form
    % training sets. If mChoosek = c, there are c unique ways to choose
    % k conditions from each side. Each of these c combinations from
    % one side can be paired against c unique combinations from the
    % other side. First, obtain all possible combinations from each side.
    trainCombos1 = nchoosek(side1, k);
    trainCombos2 = nchoosek(side2, k);
    nCombos = size(trainCombos1, 1); % nCombos = c from above inline comment
    
    % Use remaining conditions as test set. Somewhat annoyingly,
    % setdiff doesn't seem to have vectorized functionality.
    testCombos1 = NaN(nCombos, m-k);
    testCombos2 = NaN(nCombos, m-k);
    for iComb = 1:nCombos
        testCombos1(iComb,:) = setdiff(side1, trainCombos1(iComb,:));
        testCombos2(iComb,:) = setdiff(side2, trainCombos2(iComb,:));
    end
    
    % For current combination (of c combinations), test against c
    % combinations of the same size from the other side.
    for iComb1 = 1:nCombos
    for iComb2 = 1:nCombos
        
        % Get current sets of conditions (drawn from both sides)
        % that will serve as training set and test set, respectively.
        trainConds = [trainCombos1(iComb1,:) trainCombos2(iComb2,:)];
        testConds = [testCombos1(iComb1,:) testCombos2(iComb2,:)];
                     
        % Get training labels corresponding to current train set.
        trainLabels = trialLabels(ismember(trialLabels,trainConds));
        trainSet = TxN(ismember(trialLabels,trainConds),:);
        testLabels = trialLabels(ismember(trialLabels,testConds));
        testSet = TxN(ismember(trialLabels,testConds),:);
        
        % Reassign labels to 1 and 0 (instead of condition labels).
        trainLabels(ismember(trainLabels,side1)) = 1;
        trainLabels(ismember(trainLabels,side2)) = 0;
        testLabels(ismember(testLabels,side1)) = 1;
        testLabels(ismember(testLabels,side2)) = 0;
        
        % Fit classifier and test on test data.
        decoder = classifier(trainSet, trainLabels);
        [label, scores] = predict(decoder, testSet);  
        
        % Calculate accuracy and AUC.
        iDecoder = iDecoder + 1;
        accuracy(iDecoder) = ((sum(label == 1 & testLabels == 1) ...  
                              + sum(label == 0 & testLabels == 0))) ...
                             / (length(testLabels));
        [~,~,~,auc(iDecoder)] = perfcurve(testLabels, scores(:,2), 1);                          
    end
    end
end

end