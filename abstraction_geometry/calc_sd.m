function [sd,decoderPerf] = calc_sd(TxN,trialLabels,nvp)
% Calculates "shattering dimensionality" (SD) as described in Bernardi et
% al "The Geometry of Abstraction in the Hippocampus and Prefrontal
% Cortex," Cell, 2020. Returns the shattering dimensionality (average
% decoder performance over all possible dichotomies of conditions), as well
% as the decoding performance (including metrics of statistical
% significance) on all dichotomies.
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
%                       empty array, in which case the decoderPerf struct
%                       will be returned without the dichotomyConds field
%                       (see RETURNS).
%   'dropIdc'        -- 1 x n vector, where each element contains the index
%                       of neuron within the population that we wish to
%                       drop (i.e. all its entries across trials are
%                       deleted, so that the length of the second dimension
%                       of TxN overall decreases by n) before calculating
%                       CCGP.
%   'classifier'     -- Specify the function to be used as a
%                       classifier. Note that the chosen function must have
%                       'CrossVal' and 'KFold' as properties. Default is
%                       @fitclinear, which fits a linear SVM model.
%   'kFold'          -- Number of folds used to cross-validate the
%                       performance of the decoder on each dichotomy.
%   'pVal'           -- May take on values: 'two-tailed', 'left-tailed',
%                       'right-tailed', or logical false. If false, null
%                       distributions for decoder performance (accuracy and
%                       AUC) will not be computed. If one of the above
%                       string values, null distributions will be computed
%                       for decoder performance based on both accuracy and
%                       AUC (see 'nullMethod' below), and p values will be
%                       calculated accordingly. Default is 'right-tailed'.
%                       NB: this p value attaches to the performance of the
%                       decoders on the dichotomies, not the shattering
%                       dimensionality itself (see calc_fsd).
%   'permute'        -- ('trialLabels' | 'neurons'). Specify the method
%                       used to create null datasets. If 'trialLabels',
%                       each null value is created by training a
%                       classifier on the original dataset with shuffled
%                       labels. If 'neurons', each null value is created by
%                       training a classifier on a version of the dataset
%                       where each feature's (neuron's) values have been
%                       permuted but labels left unshuffled.
%   'nullInt'        -- Scalar integer specifying the interval
%                       size (as a percentage) to be calculated around the
%                       mean of the null distributions. Default = 95.
%   'nNull'          -- Scalar integer that is the number of synthetic/null
%                       datasets to generate in order to calculate
%                       statistical significance of decoding performance
%                       for each dichotomy.
%   'returnNullDist' -- (1|0, default = 0). Specify whether or not to
%                       return null distributions for decoding performance
%                       (accuracy and AUC) for all dichotomies. If so,
%                       these are stored in the accuracy and auc fields of
%                       decoderPerf (see RETURNS). This argument is ignored
%                       if 'pVal' evaluates to logical false.
%
% RETURNS
% -------
% sd             -- 1 x 1 struct with the following fields:
%   .accuracy -- Shattering dimensionality of dataset calculated as mean
%                mean decoder accuracy across all possibly dichotomies.
%   .auc      -- Shattering dimensionality of dataset calculated as mean
%                mean decoder AUC (TPR vs FPR) across all possibly
%                dichotomies.
% decoderPerf    -- 1 x 1 struct with the following fields:
%   .accuracy -- 1 x 1 struct with the following fields containing
%                information based on accuracy metric:
%       .accuracy -- nDichotomies x 1 vector, where the i_th element is the
%                    cross-validated decoding accuracy on the i_th
%                    dichotomy.
%       .p        -- nDichotomies x 1 vector, where the i_th element is the
%                    p value attached to the cross-validated decoding
%                    accuracy on the i_th dichotomy.
%       .nullInt  -- nDichotomies x 2 matrix, where each row corresponds to
%                    one dichotomy. For each row/dichotomy, the first and
%                    second column elements are the upper and lower bounds,
%                    respectively, of the specified interval around the
%                    null distribution mean; the size of this interval was
%                    specified through the 'nullInt' name value pair
%                    (default size is 95).
%       .obsStdev -- nDichotomies x 1 vector, where each element is the
%                    number of standard deviations from the mean of the
%                    i_th dichotomy's null distribution that lies the
%                    observed decoding accuracy on the i_th dichotomy.
%       .nullDist -- nDichotomies x nNull array (see 'nNull' under
%                    Name-Value Pair options above) whose i_th j_th element
%                    is the j_th draw from the accuracy null distribution
%                    for decoding the i_th dichotomy.
%   .auc            -- 1 x 1 struct with the following fields containing
%                      information based on AUC metric:
%       .accuracy -- nDichotomies x 1 vector, where the i_th element is the
%                    cross-validated decoder's AUC on the i_th dichotomy.
%       .p        -- nDichotomies x 1 vector, where the i_th element is the
%                    p value attached to the cross-validated decoder's AUC
%                    on the i_th dichotomy.
%       .nullInt  -- nDichotomies x 2 matrix, where each row corresponds to
%                    one dichotomy. For each row/dichotomy, the first and
%                    second column elements are the upper and lower bounds,
%                    respectively, of the specified interval around the
%                    null distribution mean; the size of this interval was
%                    specified through the 'nullInt' name value pair
%                    (default size is 95).
%       .obsStdev -- nDichotomies x 1 vector, where each element is the
%                    number of standard deviations from the mean of the
%                    i_th dichotomy's null distribution that lies the
%                    observed decoder's AUC on the i_th dichotomy.
%       .nullDist -- nDichotomies x nNull array (see 'nNull' under
%                    Name-Value Pair options above) whose i_th j_th element
%                    is the j_th draw from the AUC null distribution for
%                    decoding the i_th dichotomy.
%   .dichotomyConds -- An optionally returned field, dichtomyConds is an
%                      nDichotomies x nConds cell array where each row
%                      corresponds to a dichotomy. For each row
%                      (dichotomy), the first 1:nConds/2 cells contain the
%                      labels of the conditions on one side of the
%                      dichotomy and the last nConds/2+1:end cells contain
%                      the labels of the conditions on the other side of
%                      the dichotomy. If condLabels is empty (as it is by
%                      default), this field will be absent from decoderPerf.
%
% Author: Jonathan Chien 7/19/21. Last edit: 2/4/22.

arguments
    TxN
    trialLabels
    nvp.condLabels = []
    nvp.dropIdc = []
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
TxN(:,nvp.dropIdc) = [];

% Determine number of conditions, trials, and neurons.
nConds = length(unique(trialLabels));
nTrials = size(TxN, 1);
nNeurons = size(TxN, 2);

% Set combination parameter m, where we want to partition m*n objects into
% n unique groups, with n = 2 (hence a "dichotomy").
m = nConds/2;

% Set up decoderPerf to preserve order of fields if dichotomyConds added.
decoderPerf = struct('accuracy', cell (1, 1), 'auc', cell (1, 1));
   
% Get dichotomies and store corresponding condition labels in cell array
% dichotomyConds, as a field in decoderPerf.
[dichotomies,dichotomyConds] = create_dichotomies(nConds, nvp.condLabels);
nDichotomies = size(dichotomies, 1);
if ~isempty(dichotomyConds), decoderPerf.dichotomyConds = dichotomyConds; end


%% Calculate shattering dimensionality across all dichotomies

% Preallocate.
accuracy = NaN(nDichotomies, 1);
auc = NaN(nDichotomies, 1);

% Train and test decoder via cross-validation for each dichotomy. Record
% accuracy and AUC.
parfor iDichot = 1:nDichotomies
    
    % Get condition labels corresponding to either side of the current
    % dichotomy.
    side1 = dichotomies(iDichot, 1:m);
    side2 = dichotomies(iDichot, m+1:end);

    % Reassign labels to 1 and 0 (instead of condition labels).
    dichotLabels = trialLabels;
    dichotLabels(ismember(dichotLabels,side1)) = 1;
    dichotLabels(ismember(dichotLabels,side2)) = 0;
    
    % Train and test via k-fold cross-validation a classifier for current
    % dichotomy.
    cvModel = nvp.classifier(TxN, dichotLabels, 'CrossVal', 'on', 'Kfold', nvp.kFold);
    [labels, scores] = kfoldPredict(cvModel);

    % Calculate acurracy and AUC.
    accuracy(iDichot) = (sum(labels == 1 & dichotLabels == 1) ...
                        + sum(labels == 0 & dichotLabels == 0)) / nTrials;
    [~,~,~,auc(iDichot)] = perfcurve(dichotLabels, scores(:,1), 0);
end

% Calculate shattering dimensionality (sd) by averaging over accuracy and
% AUC across all dichotomies. Also store performance of decoder on each
% dichotomy.
sd.accuracy = mean(accuracy);
sd.auc = mean(auc);
decoderPerf.accuracy.accuracy = accuracy;
decoderPerf.auc.auc = auc;


%% Generate null distribution for decoding performance on all dichotomies

% Use calc_factorized_sd (which calls this function) to compare SD against
% its "null model" (a factorized representation). However, here we have the
% option to compute statistical significance for the decoding performance
% on each of the individual dichotomies.
if nvp.pVal
    
% Preallocate.
nullAccuracy = NaN(nDichotomies, nvp.nNull);
nullAuc = NaN(nDichotomies, nvp.nNull);
    
parfor iNull = 1:nvp.nNull
    
    % Generate null data.
    if strcmp(nvp.permute, 'neurons')
        % Shuffle each axis/neuron independently but leave trial label
        % unshuffled.
        nullTxN = NaN(size(TxN));
        for iNeuron = 1:nNeurons
            nullTxN(:,iNeuron) = TxN(randperm(nTrials),iNeuron);
        end
        nullTrialLabels = trialLabels;
    elseif strcmp(nvp.permute, 'trialLabels')
        % Leave unshuffled each axis/neuron but shuffle trial labels.
        nullTxN = TxN;
        nullTrialLabels = trialLabels(randperm(nTrials));
    else
        error("Invalid value for 'permute'.")
    end
    
    % Preallocate.
    currAccuracy = NaN(nDichotomies, 1);
    currAuc = NaN(nDichotomies, 1);
    
    % For current null run, obtain performance of decoders on all
    % dichotomies of null data set. All code is the same as the run on
    % empirical data above save for substitution of nullTxN for TxN.
    for iDichot = 1:nDichotomies
    
        % Get condition labels corresponding to either side of the current
        % dichotomy.
        side1 = dichotomies(iDichot, 1:m);
        side2 = dichotomies(iDichot, m+1:end);

        % Reassign labels to 1 and 0 (instead of condition labels).
        dichotLabels = nullTrialLabels;
        dichotLabels(ismember(dichotLabels,side1)) = 1;
        dichotLabels(ismember(dichotLabels,side2)) = 0;

        % Train and test via k-fold cross-validation a classifier for
        % current dichotomy.
        cvModel = nvp.classifier(nullTxN, dichotLabels, ...
                                 'CrossVal', 'on', 'Kfold', nvp.kFold);
        [labels, scores] = kfoldPredict(cvModel);
    
        % Calculate acurracy and AUC.
        currAccuracy(iDichot) = (sum(labels == 1 & dichotLabels == 1) ...
                                 + sum(labels == 0 & dichotLabels == 0)) ...
                                / nTrials;
        [~,~,~,currAuc(iDichot)] = perfcurve(dichotLabels, scores(:,1), 0);
    end
    
    % Store results of current null run after testing over all dichotomies.
    nullAccuracy(:,iNull) = currAccuracy;
    nullAuc(:,iNull) = currAuc;
end

% Option to return null distributions of accuracy and AUC (for decoding
% performance on each dichotomy).
if nvp.returnNullDist
    decoderPerf.accuracy.nullDist = nullAccuracy;
    decoderPerf.auc.nullDist = nullAuc;
end


%% Compute p values and C.I.'s for decoder performance on all dichotomies
% Again, this doesn't assess difference of SD from expected SD but assesses
% the statistical significance of decoding on each dichotomy.

% Calculate p values.
decoderPerf.accuracy.p = tail_prob(accuracy, nullAccuracy, 'type', nvp.pVal);
decoderPerf.auc.p = tail_prob(auc, nullAuc, 'type', nvp.pVal);

% Get values at 2 st.dev. in null distribution. 
decoderPerf.accuracy.nullInt = montecarlo_int(nullAccuracy, nvp.nullInt);
decoderPerf.auc.nullInt = montecarlo_int(nullAuc, nvp.nullInt);

% Calculate number of st.dev.'s away from mean that empirical value lies.
decoderPerf.accuracy.obsStdev = get_num_stdev(accuracy, nullAccuracy);
decoderPerf.auc.obsStdev = get_num_stdev(auc, nullAuc);

end

end
