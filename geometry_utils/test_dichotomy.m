function [accuracy,auc] = test_dichotomy(TxN,trialLabels,side1,side2,learner)

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
        decoder = fitclinear(trainSet, trainLabels, ...
                             'Learner', learner);
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