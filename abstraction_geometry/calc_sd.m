function [sd,decoderPerf,varargout] = calc_sd(TxN,trialLabels,nvp)
% Calculates "shattering dimensionality" (SD) as described in Bernardi et
% al "The Geometry of Abstraction in the Hippocampus and Prefrontal
% Cortex," Cell, 2020.
%
% [sd, decoderPerf] = calc_sd(TxN, trialLabels, [Name-Value Pairs])
% -----------------------------------------------------------------
% Returns "sd", a struct containing the shattering dimensionality (average
% decoder performance over all possible dichotomies of conditions), as well
% as "decoderPerf", a struct containing the performance (including metrics
% of statistical significance) on all dichotomies. Note that nConditions is
% calculated as the number of unique elements in the trialLabels vector.
%
% [sd, decoderPerf, dichotomyConds] = calc_sd(TxN, trialLabels, [Name-Value Pairs])
% ---------------------------------------------------------------------------------
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
%                   by n) before calculating CCGP. 
%   'learner'    -- Linear classification model type. Corresponds to
%                   'Learner' name-value pair in MATLAB's fitclinear.
%                   Default is 'svm', a linear SVM model.
%   'kFolds'     -- Number of folds used to cross-validate the performance
%                   of the decoder on each dichotomy.
%   'pVal'       -- May take on values: 'two-tailed', 'left-tailed',
%                   'right-tailed', or logical false. If false, null
%                   distributions for decoder performance (accuracy and
%                   AUC) will not be computed. If one of the above string
%                   values, null distributions will be computed for decoder
%                   performance based on both accuracy and AUC (see
%                   'nullMethod' below), and p values will be calculated
%                   accordingly. Default is 'two-tailed'. NB: this p value
%                   attaches to the performance of the decoders on the
%                   dichotomies, not the shattering dimensionality itself
%                   (see calculate_factorized_sd).
%   'confInt'    -- Scalar integer specifying the confidence interval size
%                   (as a percentage) to be calculated around the mean of
%                   the null distributions. Default = 95.
%   'nNull'      -- Scalar integer that is the number of synthetic/null
%                   datasets to generate in order to calculate statistical
%                   significance of decoding performance for each
%                   dichotomy.
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
%       .confInt  -- nDichotomies x 2 matrix, where each row corresponds to
%                    one dichotomy. For each row/dichotomy, the first and
%                    second column elements are the upper and lower bounds,
%                    respectively, of the confidence interval around the
%                    null distribution mean; the size of this interval was
%                    specified through the 'confInt' name value pair
%                    (default size is 95).
%       .obsStdev -- nDichotomies x 1 vector, where each element is the
%                    number of standard deviations from the mean of the
%                    i_th dichotomy's null distribution that lies the
%                    observed decoding accuracy on the i_th dichotomy.
%   .auc      -- 1 x 1 struct with the following fields containing
%                information based on AUC metric:
%       .accuracy -- nDichotomies x 1 vector, where the i_th element is the
%                    cross-validated decoder's AUC on the i_th dichotomy.
%       .p        -- nDichotomies x 1 vector, where the i_th element is the
%                    p value attached to the cross-validated decoder's AUC
%                    on the i_th dichotomy.
%       .confInt  -- nDichotomies x 2 matrix, where each row corresponds to
%                    one dichotomy. For each row/dichotomy, the first and
%                    second column elements are the upper and lower bounds,
%                    respectively, of the confidence interval around the
%                    null distribution mean; the size of this interval was
%                    specified through the 'confInt' name value pair
%                    (default size is 95).
%       .obsStdev -- nDichotomies x 1 vector, where each element is the
%                    number of standard deviations from the mean of the
%                    i_th dichotomy's null distribution that lies the
%                    observed decoder's AUC on the i_th dichotomy.
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
% Author: Jonathan Chien Version 1.1. 7/19/21 Last edit: 7/22/21.


arguments
    TxN
    trialLabels
    nvp.condLabels = []
    nvp.dropIdc = []
    nvp.learner = 'svm'
    nvp.kFolds (1,1) = 5
    nvp.pVal = 'two-tailed'
    nvp.confInt (1,1) = 95
    nvp.nNull (1,1) = 1000
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
   
% Get dichotomies and store corresponding condition labels in cell array
% dichotomyConds.
[dichotomies,dichotomyConds] = create_dichotomies(nConds, nvp.condLabels);
nDichotomies = size(dichotomies, 1);
varargout{1} = dichotomyConds;
if nargout == 3 && isempty(nvp.condLabels)
    warning(['Dichotomy labels requested but no condition labels were ' ...
             'supplied. dichotomyConds will be returned as an empty array.'])  
end


%% Calculate shattering dimensionality across all dichotomies

% Preallocate.
accuracy = NaN(nDichotomies, 1);
auc = NaN(nDichotomies, 1);

% Train and test decoder via cross-validation for each dichotomy. Record
% accuracy and AUC.
for iDichot = 1:nDichotomies
    
    % Get condition labels corresponding to either side of the current
    % dichotomy.
    side1 = dichotomies(iDichot, 1:m);
    side2 = dichotomies(iDichot, m+1:end);

    % Reassign labels to 1 and 0 (instead of condition labels).
    dichotLabels = trialLabels;
    dichotLabels(ismember(dichotLabels,side1)) = 1;
    dichotLabels(ismember(dichotLabels,side2)) = 0;
    
    % Initialize for k-fold CV. allTestLabels is used to calculate AUC
    % after iterating across all k folds.
    correctPred = 0;
    allTestLabels = NaN(nTrials, 1);
    predLabels = NaN(nTrials, 1);
    predScores = NaN(nTrials, 2);
    foldSize = nTrials / nvp.kFolds;
    assert(mod(nTrials,nvp.kFolds)==0, ...
        'nTrials must be evenly divisble by kFolds.')
    cvIndices = crossvalind('Kfold', nTrials, nvp.kFolds);
    
    % Cross-validate.
    for k = 1:nvp.kFolds
        
        % Designate train and test sets. 
        trainLabels = dichotLabels(cvIndices~=k);
        trainSet = TxN(cvIndices~=k,:);
        testLabels = dichotLabels(cvIndices==k);
        testSet = TxN(cvIndices==k,:);
        allTestLabels((k-1)*foldSize+1 : k*foldSize) = testLabels;
        
        % Fit classifier and test on test data.
        decoder = fitclinear(trainSet, trainLabels, 'Learner', nvp.learner);
        [label, scores] = predict(decoder, testSet);  
        
        % Accumulate correct classifications to calculate accuracy after
        % iterating through all k folds. Correct predictions = TP + TN.
        correctPred = correctPred + (sum(label == 1 & testLabels == 1) + ...
                                     sum(label == 0 & testLabels == 0));
                                 
        % Save labels and scores from current fold to calculate accuracy
        % and AUC when finished iterating across k folds.
        predLabels((k-1)*foldSize+1 : k*foldSize) = label;
        predScores((k-1)*foldSize+1 : k*foldSize,:) = scores;
    end          
    
    % Calculate accuracy and AUC after iterating over k folds.
    accuracy(iDichot) = correctPred / nTrials;
    [~,~,~,auc(iDichot)] = perfcurve(allTestLabels, predScores(:,2), 1);
end

% Calculate shattering dimensionality (sd) by averaging over accuracy and
% AUC across all dichotomies. Also store performance of decoder on each
% dichotomy.
sd.accuracy = mean(accuracy);
sd.auc = mean(auc);
decoderPerf.accuracy.accuray = accuracy;
decoderPerf.auc.auc = auc;


%% Generate null distribution for decoding performance on all dichotomies

% Use calc_expected_sd (which calls this function) to compared SD against
% its "null model" (a factorized representation). However, here we have the
% option to compute statistical significant for the decoding performance on
% each of the individual dichotomies.
if nvp.pVal
    
% Preallocate.
nullAccuracy = NaN(nDichotomies, nvp.nNull);
nullAuc = NaN(nDichotomies, nvp.nNull);
    
parfor iNull = 1:nvp.nNull
    
    % Generate null data set by shuffling each axis/neuron independently.
    nullTxN = NaN(size(TxN));
    for iNeuron = 1:nNeurons
        nullTxN(:,iNeuron) = TxN(randperm(nTrials),iNeuron);
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
        dichotLabels = trialLabels;
        dichotLabels(ismember(dichotLabels,side1)) = 1;
        dichotLabels(ismember(dichotLabels,side2)) = 0;

        % Initialize for k-fold CV. allTestLabels is used to calculate AUC
        % after iterating across all k folds.
        correctPred = 0;
        allTestLabels = NaN(nTrials, 1);
        predLabels = NaN(nTrials, 1);
        predScores = NaN(nTrials, 2);
        foldSize = nTrials / nvp.kFolds;
        assert(mod(nTrials,nvp.kFolds)==0, ...
            'nTrials must be evenly divisble by kFolds.')
        cvIndices = crossvalind('Kfold', nTrials, nvp.kFolds);

        % Cross-validate.
        for k = 1:nvp.kFolds

            % Designate train and test sets. 
            trainLabels = dichotLabels(cvIndices~=k);
            trainSet = nullTxN(cvIndices~=k,:);
            testLabels = dichotLabels(cvIndices==k);
            testSet = nullTxN(cvIndices==k,:);
            allTestLabels((k-1)*foldSize+1 : k*foldSize) = testLabels;

            % Fit classifier and test on test data.
            decoder = fitclinear(trainSet, trainLabels, 'Learner', nvp.learner);
            [label, scores] = predict(decoder, testSet);  

            % Accumulate correct classifications to calculate accuracy after
            % iterating through all k folds. Correct predictions = TP + TN.
            correctPred = correctPred + (sum(label == 1 & testLabels == 1) + ...
                                         sum(label == 0 & testLabels == 0));

            % Save labels and scores from current fold to calculate accuracy
            % and AUC when finished iterating across k folds.
            predLabels((k-1)*foldSize+1 : k*foldSize) = label;
            predScores((k-1)*foldSize+1 : k*foldSize,:) = scores;
        end          

        % Calculate accuracy and AUC after iterating over k folds.
        currAccuracy(iDichot) = correctPred / nTrials;
        [~,~,~,currAuc(iDichot)] = perfcurve(allTestLabels, predScores(:,2), 1);
    end
    
    % Store results of current null run after testing over all dichotomies.
    nullAccuracy(:,iNull) = currAccuracy;
    nullAuc(:,iNull) = currAuc;
end


%% Compute p values and C.I.'s for decoder performance on all dichotomies
% Again, this doesn't assess difference of SD from expected SD but assesses
% the statistical significance of decoding on each dichotomy.

% Calculate p values.
decoderPerf.accuracy.p = tail_prob(accuracy, nullAccuracy, 'type', nvp.pVal);
decoderPerf.auc.p = tail_prob(auc, nullAuc, 'type', nvp.pVal);

% Get values at 2 st.dev. in null distribution. 
decoderPerf.accuracy.confInt = montecarlo_conf_int(nullAccuracy, nvp.confInt);
decoderPerf.auc.confInt = montecarlo_conf_int(nullAuc, nvp.confInt);

% Calculate number of st.dev.'s away from mean that empirical value lies.
decoderPerf.accuracy.obsStdev = get_num_stdev(accuracy, nullAccuracy);
decoderPerf.auc.obsStdev = get_num_stdev(auc, nullAuc);

end

end