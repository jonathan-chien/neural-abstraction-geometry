function [sd,performance,parsedParams] = calc_sd(TxN,condLabels,nvp)
% Calculates "shattering dimensionality" (SD) as described in Bernardi et
% al "The Geometry of Abstraction in the Hippocampus and Prefrontal
% Cortex," Cell, 2020. Returns the shattering dimensionality (average
% decoder performance over all possible dichotomies of conditions), as well
% as the decoding performance (including metrics of statistical
% significance) on all dichotomies.
%
% PARAMETERS
% ----------
% TxN        -- T-by-N matrix, where T = nTrials and N = nNeurons. The
%               i_th row j_th column element is the firing rate of the j_th
%               neuron on the i_th single trial. All trials (rows of TxN)
%               belong to one of nConds conditions with nTrials >> nConds.
%               Note that each of the nConds conditions is represented
%               nTrials / nConds times among the rows of TxN.
% condLabels -- Vector of length = nTrials, where each element is a
%               positive integer condition label corresponding to a row
%               (single trial) of TxN. The number of conditions is
%               calculated as the number of unique elements in the
%               condLabels vector.
% Name-Value Pairs (nvp)
%   'dropInd'        -- 1 x n vector, where each element contains the index
%                       of neuron within the population that we wish to
%                       drop (i.e. all its entries across trials are
%                       deleted, so that the length of the second dimension
%                       of TxN overall decreases by n) before calculating
%                       CCGP.
%   'condNames'     -- 1 x nConds cell array, where each cell contains the
%                       name/description of a condition. Default is an
%                       empty array, in which case the decoderPerf struct
%                       will be returned without the dichotomyConds field
%                       (see RETURNS).
%   'decoderParams'  -- A scalar struct that has as its fields the
%                       name-value pairs accepted by multiset_decoder.m. If
%                       empty, default params will be set (see local
%                       function parse_inputs).
%
% RETURNS
% -------
% sd             -- 1 x 1 struct with the following fields:
%   .accuracy  -- Shattering dimensionality of dataset calculated as mean
%                 mean decoder accuracy across all possibly dichotomies.
%   .precision -- Shattering dimensionality of dataset calculated as mean
%                 mean decoder precision across all possibly dichotomies.
%   .recall    -- Shattering dimensionality of dataset calculated as mean
%                 mean decoder recall across all possibly dichotomies.
%   .fmeasure  -- Shattering dimensionality of dataset calculated as mean
%                 mean decoder fmeasure across all possibly dichotomies.
%   .aucroc    -- Shattering dimensionality of dataset calculated as mean
%                 mean decoder AUC ROC (TPR vs FPR) across all possible
%                 dichotomies.
%   .aucpr     -- Shattering dimensionality of dataset calculated as mean
%                 mean decoder AUC PR (recall vs precision) across all
%                 possible dichotomies.
% performance -- Scalar struct with the following fields:
%   .accuracy    -- nDichotomies x 1 vector whose i_th element is the the
%                   micro-averaged accuracy across classes (for an
%                   aggregation across folds, this is the sum of true
%                   positive for each class, divided by the total number of
%                   observations; this is also equivalent to micro-averaged
%                   precision, recall, and f-measure), averaged across
%                   repetitions, for the i_th dichotomy/dataset.
%   .balaccuracy -- nDichotomies x 1 vector whose i_th element is the the
%                   balanced accuracy across classes (average of recall for
%                   each class), averaged across repetitions, for the i_th
%                   dichotomy/dataset.
%   .precision   -- nDichotomies x nClasses matrix whose i_th j_th element
%                   is the precision (number of true positives divided by
%                   number of predicted positives) for the j_th class as
%                   positive, averaged across repetitions, in the i_th
%                   dichotomy.
%   .recall      -- nDichotomies x nClasses matrix whose i_th j_th element
%                   is the precision (number of true positives divided by
%                   number of actual positives) for the j_th class as
%                   positive, averaged across repetitions, in the i_th
%                   dichotomy.
%   .fmeasure    -- nDichotomies x nClasses matrix of f-measure scores
%                   whose i_th j_th element is the f-measure for the j_th
%                   class as positive, averaged across repetitions, in the
%                   i_th dichotomy.
%   .aucroc      -- nDichotomies x nClasses matrix of AUC ROC values whose
%                   i_th j_th element is the AUC ROC for the j_th class as
%                   positive, averaged across repetitions, in the i_th
%                   dichotomy.
%   .aucpr       -- nDichotomies x nClasses matrix of AUC PR
%                   (precision-recall) values whose i_th j_th element is
%                   the AUC PR for the j_th class as positive, averaged
%                   across repetitions, in the i_th dichotomy.
%   .confMat     -- Optionally returned field (if 'saveConfMat' = true)
%                   consisitng of an nDichotomies x nReps x nClasses x
%                   nClasses array where the (i,j,:,:) slice contains the
%                   nClasses x nClasses confusion matrix (rows: true class,
%                   columns: predicted class) for the j_th repetition in
%                   the i_th dichotomy.
%   .labels      -- Optionally returned field (if 'saveLabels' = true and
%                   'oversampleTest' = false) consisting of an nDichotomies
%                   x nReps x nObservations numeric array, where the i_th
%                   row contains the predicted labels (aggregated across
%                   folds) for the i_th repetition.
%   .scores      -- Optionally returned field (if 'saveScores' = true and
%                   'oversampleTest' = false) consisting of an nDichotomies
%                   x nReps x nObservations x nClasses numeric array, where
%                   the (i,j,:,:) slice contains the nObservations x
%                   nClasses matrix for the j_th repetition in the i_th
%                   dichotomy, and the k_th l_th element (of this slice) is
%                   the score for the k_th observation being classified
%                   into the l_th class.
%   .sig         -- Scalar struct with the following fields (all individual
%                   null performance values, from whose aggregate the
%                   following measures are calculated are generated from
%                   only one repetition of a given permutation).
%       .p        -- Scalar struct with the following fields (note: these p
%                    values are calculated using an observed performance
%                    values that are the average across repetitions).
%           .accuracy    -- nDichotomies x 1 vector of p values for
%                           micro-averaged accuracy.
%           .balaccuracy -- nDichotomies x 1 vector of p values for
%                           balanced accuracy.
%           .precision   -- nDichotomies x nClasses matrix of p values for 
%                           precision.
%           .recall      -- nDichotomies x nClasses matrix of p values for 
%                           recall. 
%           .fmeasure    -- nDichotomies x nClasses matrix of p values for 
%                           f-measure.
%           .aucroc      -- nDichotomies x nClasses matrix of p values for 
%                           AUC ROC.
%           .aucpr       -- nDichotomies x nClasses matrix of p values for 
%                           AUC PR.
%       .nullInt  -- Scalar struct with the following fields (size of
%                    interval dictated by the 'nullInt' name-value pair.
%           .accuracy    -- nDichotomies x 2 matrix whose i_th row has as
%                           its 1st and 2nd elements the lower and upper
%                           bounds of the interval on the accuracy null
%                           distribution for the i_th dichotomy.
%           .balaccuracy -- nDichotomies x 2 matrix whose i_th row has as
%                           its 1st and 2nd elements the lower and upper
%                           bounds of the interval on the balanced accuracy
%                           null distribution for the i_th dichotomy.
%           .precision   -- nDichotomies x nClasses x 2 array where the
%                           (i,:,:) slice is a matrix whose j_th row has as
%                           its 1st and 2nd elements the lower and upper
%                           interval bounds on the precision null
%                           distribution, where the j_th class is positive,
%                           in the i_th dichotomy.
%           .recall      -- nDichotomies x nClasses x 2 array where the
%                           (i,:,:) slice is a matrix whose j_th row has as
%                           its 1st and 2nd elements the lower and upper
%                           interval bounds on the recall null
%                           distribution, where the j_th class is positive,
%                           in the i_th dichotomy.
%           .fmeasure    -- nDichotomies x nClasses x 2 array where the
%                           (i,:,:) slice is a matrix whose j_th row has as
%                           its 1st and 2nd elements the lower and upper
%                           interval bounds on the f-measure null
%                           distribution, where the j_th class is positive,
%                           in the i_th dichotomy.
%           .aucroc      -- nDichotomies x nClasses x 2 array where the
%                           (i,:,:) slice is a matrix whose j_th row has as
%                           its 1st and 2nd elements the lower and upper
%                           interval bounds on the AUC ROC null
%                           distribution, where the j_th class is positive,
%                           in the i_th dichotomy.
%           .aucpr       -- nDichotomies x nClasses x 2 array where the
%                           (i,:,:) slice is a matrix whose j_th row has as
%                           its 1st and 2nd elements the lower and upper
%                           interval bounds on the AUC PR null
%                           distribution, where the j_th class is positive,
%                           in the i_th dichotomy. 
%       .obsStdev -- Scalar struct with the following fields:
%           .accuracy    -- nDichotomies x 1 vector whose i_th element is
%                           the number of standard deviations from the mean
%                           of the null accuracy distribution that the
%                           observed accuracy (averaged across repetitions)
%                           lies, for the i_th dichotomy.
%           .balaccuracy -- nDichotomies x 1 vector whose i_th element is
%                           the number of standard deviations from the mean
%                           of the null balanced accuracy distribution that
%                           the observed balanced accuracy (averaged across
%                           repetitions) lies, for the i_th dichotomy.
%           .precision   -- nDichotomies x nClasses matrix whose i_th j_th
%                           element is the number of standard deviations
%                           from the mean of the null precision
%                           distribution (for the j_th class as positive)
%                           that the observed precision (averaged across
%                           repetitions) lies, for the i_th dichotomy.
%           .recall      -- nDichotomies x nClasses matrix whose i_th j_th
%                           element is the number of standard deviations
%                           from the mean of the null recall distribution
%                           (for the j_th class as positive) that the
%                           observed recall (averaged across repetitions)
%                           lies, for the i_th dichotomy.
%           .fmeasure    -- nDichotomies x nClasses matrix whose i_th j_th
%                           element is the number of standard deviations
%                           from the mean of the null f-measure
%                           distribution (for the j_th class as positive)
%                           that the observed f-measure (averaged across
%                           repetitions) lies, for the i_th dichotomy.
%           .aucroc      -- nDichotomies x nClasses matrix whose i_th j_th
%                           element is the number of standard deviations
%                           from the mean of the null AUC ROC distribution
%                           (for the j_th class as positive) that the
%                           observed AUC ROC (averaged across repetitions)
%                           lies, for the i_th dichotomy.
%           .aucpr       -- nDichotomies x nClasses matrix whose i_th j_th
%                           element is the number of standard deviations
%                           from the mean of the null AUC PR distribution
%                           (for the j_th class as positive) that the
%                           observed AUC PR (averaged across repetitions)
%                           lies, for the i_th dichotomy.
%   .dichotomyConds -- An optionally returned field, dichtomyConds is an
%                      nDichotomies x nConds cell array where each row
%                      corresponds to a dichotomy. For each row
%                      (dichotomy), the first 1:nConds/2 cells contain the
%                      labels of the conditions on one side of the
%                      dichotomy and the last nConds/2+1:end cells contain
%                      the labels of the conditions on the other side of
%                      the dichotomy. If condNames is empty (as it is by
%                      default), this field will be absent from decoderPerf.
% parsedParams -- Scalar struct that has its fields all the parsed params
%                 passed to multiset_decoder.m.
%
% Author: Jonathan Chien 7/19/21. Last edit: 3/16/22.

arguments
    TxN
    condLabels
    nvp.condNames = []
    nvp.dropInd = []
    nvp.decoderParams = []
    nvp.classifier = @fitclinear
    nvp.kFold (1,1) = 5
    nvp.pVal = 'right-tailed'
    nvp.permute = 'trialLabels'
    nvp.nullInt (1,1) = 95
    nvp.nNull (1,1) = 1000
    nvp.returnNullDist = false
end

%% Process inputs and prepare for decoding runs

% Option to drop neurons if desired.
TxN(:,nvp.dropInd) = [];

% Determine number of conditions, trials, and neurons.
nConds = length(unique(condLabels));
nTrials = size(TxN, 1);

% Set combination parameter m, where we want to partition m*n objects into
% n unique groups, with n = 2 (hence a "dichotomy").
m = nConds/2;
   
% Get dichotomies and store corresponding condition labels in cell array
% dichotomyConds, as a field in decoderPerf.
[dichotomies,dichotomyConds] = create_dichotomies(nConds, nvp.condNames);
nDichotomies = size(dichotomies, 1);



%% Calculate shattering dimensionality across all dichotomies

% Prepare class labels for each dichotomy.
dichotLabels = NaN(nTrials, nDichotomies);
for iDichot = 1:nDichotomies
    % Get condition labels corresponding to either side of the current
    % dichotomy.
    side1 = dichotomies(iDichot, 1:m);
    side2 = dichotomies(iDichot, m+1:end);

    % Reassign labels to 1 and 0 (instead of condition labels).
    dichotLabels(ismember(condLabels,side1),iDichot) = 1;
    dichotLabels(ismember(condLabels,side2),iDichot) = 0;
end

% Parse params for decoding. If passed in empty, default params will be
% used. Parsed params will also be returned as parsedParams.
if isempty(nvp.decoderParams), nvp.decoderParams = struct('a', []); end
parsedParams = parse_inputs(nvp.decoderParams);
    
% Test decoding on same dataset but with different class labels (for
% different dichotomies).
performance ...
    = multiset_decoder(TxN, dichotLabels, ...
                       'classifier',  parsedParams.classifier, ...
                       'kFold', parsedParams.kFold, ... 
                       'nCvReps', parsedParams.nCvReps, ...
                       'cvFun', parsedParams.cvFun, ...
                       'oversampleTrain', parsedParams.oversampleTrain, ...
                       'oversampleTest', parsedParams.oversampleTest, ...
                       'condLabels', condLabels, ...
                       'nResamples', parsedParams.nResamples, ...                         
                       'permute', parsedParams.permute, ...
                       'nPerms', parsedParams.nPerms, ...
                       'saveLabels', parsedParams.saveLabels, ...
                       'saveScores', parsedParams.saveScores, ...
                       'saveConfMat', parsedParams.saveConfMat, ...
                       'pval', parsedParams.pval, ...
                       'nullInt', parsedParams.nullInt, ...
                       'concatenate', true);    

% Calculate shattering dimensionality as average across dichotomies.
perfMetrics = {'accuracy', 'balaccuracy', 'precision', 'recall', ...
               'fmeasure', 'aucroc', 'aucpr'};
for iField = 1:length(perfMetrics)
    metric = perfMetrics{iField};
    sd.(metric) = mean(performance.(metric), 1);
end

% If condition names were supplied, return the labeled dichotomies as a
% field of performance.
if ~isempty(dichotomyConds), performance.dichotomyConds = dichotomyConds; end


end


% --------------------------------------------------
function parsedParams = parse_inputs(passedParams)
% Each field in the defaultParams struct is checked against the fields of
% the passed in params (passedParams). If a match is found, the value of
% the field in passedParams is assigned to the field of the same name in
% parsedParams. If no match is found, the value of the field from
% defaultParams is used instead. Note that any extraneous fields in
% passedParams will thus be ignored. Note as well that the condLabels field
% for multiset_decoder is not included here, as the condLabels argument
% for calc_sd is used instead.

% Default options.
defaultParams = struct('classifier', @fitclinear, ...
                       'kFold', 10, 'nCvReps', 100, 'cvFun', 2, ...
                       'oversampleTrain', 'byCond', ...
                       'oversampleTest', 'byCond', ...
                       'nResamples', 100, 'permute', 'labels', ...
                       'nPerms', 1000, 'saveLabels', false, ...
                       'saveScores', false, 'saveConfMat', false, ...
                       'pval', 'right-tailed', 'nullInt', 95);

% Create matching parsedParms struct with same fields as default, but all
% empty.
fnamesDefault = fieldnames(defaultParams);
for iField = 1:length(fnamesDefault)
    parsedParams.(fnamesDefault{iField}) = [];
end

% Get all fields of passed in params struct.
fnamesPassed = fieldnames(passedParams);

% For each field in parsedParams, use passed in value if a matching passed
% in value exists, else use default value.
for iField = 1:length(fnamesDefault)
    field = fnamesDefault{iField};

    if cell2mat(strfind(fnamesPassed, field))
        parsedParams.(field) = passedParams.(field);
    else
        parsedParams.(field) = defaultParams.(field);
    end
end

end
