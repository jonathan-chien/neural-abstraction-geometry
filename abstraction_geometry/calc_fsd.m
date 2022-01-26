function factorized = calc_fsd(sizes,targetCcgp,nvp)
% Calculates "shattering dimensionality" (SD) for a factorized geometry as
% described in Bernardi et al "The Geometry of Abstraction in the
% Hippocampus and Prefrontal Cortex," Cell, 2020.
%
% PARAMETERS
% ----------
% sizes      -- 3-vector whose elements specify the number of trials,
%               neurons, and conditions, respectively, to be used in
%               modeling the factorized model.
% targetCcgp -- Vector whose elements contain the target CCGPs which we
%               will try to achieve by tuning the hypercuboid representing
%               the factorized model. For 8 conditions, this vector may
%               have 1 - 3 elements; for 4 conditions, it may have 1 or 2
%               elements.
% Name-Value Pairs (nvp)
%   'disFactorVals'  -- 2D Array of values to be tested over when
%                       tuning lengths of the hypercuboid. Each row
%                       corresponds to one dichotomy and to a set of
%                       factors for tuning one side of the hypercuboid (the
%                       direction orthogonal to a separating hyperplane for
%                       that dichotomy). The maximum number of
%                       rows/decodable dichotomies is 2 for the 4 condition
%                       case and 3 for the 8 condition case (for m
%                       conditions as the vertices of an m-dimensional
%                       hyercube, the instrinsic dimensionality of the
%                       hypercube (and thus the dimension of the embedding
%                       vector space if the hypercube is centered at the
%                       origin) is log2(m) = n; in such an n-dimensional
%                       space, there are n translational degrees of
%                       freedom, and linear separation requires us to be
%                       able to move from one class to another in a single
%                       direction; hence n linear separations are
%                       possible). Note however, that the number of models
%                       tested is the number of tuning values raised to the
%                       power of the number of dichotomies, so overly
%                       fine-grained searches may come with a very high
%                       computational cost, especially in the 8 condition
%                       case.
%   'tuneWith'       -- (string: 'accuracy'| 'auc') Specify which
%                       classification performance metric should be
%                       compared against targetCcgp (this should correspond
%                       to the same performance metric as targetCcgp).
%   'lossFunction'   -- (string: 'sae'|'sse'|'rss'|'mse') Specify
%                       the function used to calculate error between a
%                       tuned hypercuboid's CCGPs and the passed in target
%                       CCGPs.
%   'tuneOnTrials'   -- (scalar|false) Specify the number of trials to be
%                       added to the vertices of the hypercube when tuning.
%                       Set to false to tune using classifiers trained on
%                       vertices of hypercube alone (not recommended for
%                       nConditions = 4, since classifiers will then train
%                       on two observations and usually have accuracy of 0
%                       or 1 (and CCGP is the average over only 4
%                       classifier's performances). Note well that this is
%                       is not related to the number of trials added to the
%                       vertices of the hypercube (after tuning) when
%                       calculating the SD and final, full CCGP (rather,
%                       this number of trials per vertex is inferred as the
%                       ratio of the first and third element of the input
%                       arg "sizes".   
%   'trialVar'       -- (vector or scalar) Pass a range of variances of the
%                       gaussian distribution from which we make draws to
%                       construct a point cloud added to each vertex.
%                       (corresponding to simulated single trials). If
%                       'tuneOnTrials' ~= false, the function will test
%                       all variance values during the tuning step and
%                       select the optimal variance to apply to the point
%                       cloud surrounding vertices of the final, tuned
%                       hypercube during calculation of SD and the final,
%                       full CCGP. Note that if 'tuneOnTrials' = false,
%                       the first element of a vector argument is used by
%                       default (so it may be in this case preferrable to
%                       instead pass in a scalar corresponding to the
%                       specific variance desired in the point cloud
%                       surrounding the vertices of the final, tuned
%                       hypercuboid to be tested).
%   'kFolds'         -- (scalar) Once the length of the hypercube is tuned
%                       and gaussian noise added, the shattering
%                       dimenisionality of the factorized model is
%                       calculated via the calc_sd function. This
%                       name-value pair specifies the number of folds to
%                       use when cross validating the decoder performance
%                       on each of the dichotomies within that routine
%                       (briefly, SD is the average decoding performance
%                       over all dichotomies. See calc_sd documentation for
%                       more information).
%   'learner'        -- (string) Once the length of the hypercube is tuned
%                       and gaussian noise added, the shattering
%                       dimenisionality and CCGP of the factorized model
%                       are calculated via the calc_sd function and
%                       test_ccgp function, respectively. For both of those
%                       routines, specify the linear classification model
%                       type with this name-value pair. Corresponds to
%                       'Learner' name-value pair in MATLAB's fitclinear.
%                       Default = 'svm'.
%   'pVal'           -- ('left-tailed'|'right-tailed'|'two-tailed'|false
%                       (default)) Once the length of the hypercube is
%                       tuned and guassian noise added, the CCGP of this
%                       model is calculated via the test_ccgp function. For
%                       this routine, pass a string specifying the type of
%                       test or set to logical false to suppress
%                       calculation of signficance for the CCGP of the
%                       dichotomies in the tuned factorized model.
%   'nNull'          -- (scalar: default = 1000) Once the length of the
%                       hypercube is tuned and gaussian noise added, the
%                       CCGP of this factorized model is calculated via the
%                       test_ccgp function. For this routine, specify the
%                       number of null datasets used to generate the p
%                       values attached to the CCGP values returned.
%   'nullMethod'     -- (string: 'permutation'|'geometric') Once the length
%                       of the hypercube is tuned and gaussian noise added,
%                       the CCGP of this factorized model is calculated via
%                       the test_ccgp function. For this routine, specify
%                       the method used to generate the null datasets. (see
%                       test_ccgp documentation for more information).
%   'returnNullDist' -- (logical: 1|0 (default)). Specify whether or not
%                       to return the null distributions for accuracy and
%                       auc (for each dichotomy) in the ccgp field (see
%                       RETURNS below; also see the 'returnNullDist'
%                       Name-Value Pair in test_ccgp for more information).
%                       Again, this argument will be ignored if 'pVal'
%                       evaluates to logical false.
%                               
% RETURNS
% -------
% factorized -- 1 x 1 struct with the following fields:
%   .sd          -- 1 x 1 struct containing shattering dimensionality
%                   information about the tuned factorized model (this is
%                   the output of the calc_sd function when called on the
%                   factorized model).
%   .decoderPerf -- 1 x 1 struct containing information about decoding
%                   performance (accuracy and auc) for each dichotomy of
%                   the factorized model (see decoderPerf under RETURNS in
%                   the documentation for calc_sd for more information).
%                   Currently, there is no option to compute significance
%                   measures for these performances.
%   .ccgp        -- 1 x 1 struct containing CCGP information about the
%                   tuned factorized model (this is the output of the
%                   test_ccgp function when called on the factorized
%                   model).
%
% Author: Jonathan Chien Version 1.0. 7/24/21 Last edit: 8/24/21.


arguments
    sizes % 3-vector = [nTrials nNeurons nConds]
    targetCcgp 
    nvp.disFactorVals = 1 : 0.1 : 2
    nvp.tuneWith string = 'accuracy' % accuracy or AUC
    nvp.lossFunction = 'sae'
    nvp.tuneOnTrials = false
    nvp.trialVar = 1 % controls variance around each vertex of the hypercube
    nvp.kFolds = 5 % for testing sd of factorized model
    nvp.learner = 'svm' % for testing sd of factorized model
    nvp.pVal = false % This is for CCGP of factorized model
    nvp.nNull = 1000 
    nvp.nullMethod = 'geometric'
    nvp.returnNullDist = false
end


%% Parse and check arguments

% Determine number of trials, neurons, conditions, and trials per
% condition, as well as number of factorized models we will need to tune
% and test (nModels).
nTrials = sizes(1);
nNeurons = sizes(2);
nConds = sizes(3);
nTrialsPerCond = nTrials / nConds;
assert(mod(nTrials,nConds)==0, 'nTrials must be evenly divisible by nConds.')
nDichotToTune = length(targetCcgp);
nDisFactorVals = length(nvp.disFactorVals);
nVarVals = length(nvp.trialVar);
if ~nvp.tuneOnTrials && nVarVals > 1
    warning(["A range of trial variances was supplied, but tuning was " ...
             "requested based on vertices only. The first variance value " ...
             "will be used by default when constructing the final tuned " ...
             "hypercube, so it is recommended to pass only the desired " ...
             "point cloud variance as a scalar."])
    nvp.trialVar = nvp.trialVar(1);
    nVarVals = 1;
end

% Check for potentially problematic values for nConds.
if nConds ~= 8 && nConds ~= 4
    warning('This function currently only supports nConds = 4 or nConds = 8.')
end

% Ensure that the user has not requested the tuning of too many dichotomies.
if nConds == 4
    assert(nDichotToTune <= 2, ...
           ['For a factorized model of 4 conditions, only 2 dichotomies are ' ...
            'decodable.']);
elseif nConds == 8
    assert(nDichotToTune <= 3, ...
           ['For a factorized model of 8 conditions, only 3 dichotomies are ' ...
            'decodable.']);
end

% Expand the tuning factors into an array whose first dim size matches the
% number of dichotomies we would like to tune. Then pad with rows of zeros
% so that the first dim size = 3.
expandedDisFactorVals = repmat(nvp.disFactorVals, nDichotToTune, 1);
if nDichotToTune < 3
    expandedDisFactorVals = [expandedDisFactorVals; ...
                             zeros(3-nDichotToTune, nDisFactorVals)];
end


%% Tune hypercuboid

% If user opted to add simulated trials cloud around each vertex of the
% hypercube, expand the training labels to match.
if nvp.tuneOnTrials
    tuningTrialLabels = repelem((1:nConds)', nvp.tuneOnTrials);
else
    tuningTrialLabels = (1:nConds)';
end

% Preallocate (same container size regardless of condition number due to
% 'omitnan' in min function).
ccgpLoss = NaN(nDisFactorVals, nDisFactorVals, nDisFactorVals, nVarVals);

% Try all combinations of different values for displacement, which tunes
% the length of the hypercube. 
parfor iFactor1 = 1:nDisFactorVals
    for iFactor2 = 1:nDisFactorVals
        for iFactor3 = 1:nDisFactorVals
            for iVar = 1:nVarVals
            
                % Obtain current set of displacement scaling factors.
                currDisFactors = [expandedDisFactorVals(1,iFactor1) ...
                                  expandedDisFactorVals(2,iFactor2) ...
                                  expandedDisFactorVals(3,iFactor3)];
                if nConds == 4, currDisFactors(3) = []; end
        
                % Using current displacement factors, get vertices of
                % cuboid embedded in N-space, with N = nNeurons. We don't
                % apply sqrt to nvp.trialVar here, because sqrt will be
                % applied in embed_hypercube.
                [hypercuboid, dichotomies] ...
                    = embed_hypercuboid(nNeurons, nConds, currDisFactors, ...
                                        'addTrials', nvp.tuneOnTrials, ...
                                        'trialVar', nvp.trialVar(iVar)); 
                
                % Calculate CCGPs of dichotomies for which the current
                % hypercube was tuned. For speed, CCGP is calculated only
                % for these specific dichotomies.
                ccgp = test_ccgp(hypercuboid, tuningTrialLabels, ...
                                 'dichotomies', dichotomies, 'pVal', false);
                switch nvp.tuneWith
                    case 'accuracy'
                        ccgpLoss(iFactor1,iFactor2,iFactor3,iVar) ...
                            = calc_loss(targetCcgp, ...
                                        ccgp.accuracy.ccgp(1:nDichotToTune), ...
                                        nvp.lossFunction);
                    case 'auc'
                        ccgpLoss(iFactor1,iFactor2,iFactor3,iVar) ...
                            = calc_loss(targetCcgp, ...
                                        ccgp.auc.ccgp(1:nDichotToTune), ...
                                        nvp.lossFunction);
                    otherwise
                        error("Invalid value for 'tuneWith'.")
                end
            end
        
            % Breaking condition.
            if nConds == 4 || nDichotToTune < 3, break; end
        end
    
        % Breaking condition.
        if nDichotToTune == 1, break; end
    end
end


%% Construct optimal tuned hypercuboid and calculate CCGP and SD

% Select displacement factors leading to smallest error between target
% CCGPs and constructed CCGP(s). 
[~,iMin] = min(ccgpLoss, [], 'all', 'omitnan', 'linear');
[iFactor1Opt,iFactor2Opt,iFactor3Opt,iVarOpt] ...
    = ind2sub([repmat(nDisFactorVals, 1, 3) nVarVals], iMin);
optDisFactors = [expandedDisFactorVals(1,iFactor1Opt) ...
                 expandedDisFactorVals(2,iFactor2Opt) ...
                 expandedDisFactorVals(3,iFactor3Opt)];
if nConds == 4, optDisFactors(3) = []; end

% If one of the optimal parameters (including trial variance) was the
% largest or smallest provided, warn user that search grid may need to be
% shifted/modified to better cover loss basin.
if any(optDisFactors == nDisFactorVals) 
    warning(['The optimal geometric structure found occured at one of the ' ...
             'upper edges of the supplied search grid. Consider shifting ' ...
             'up the range of the displacmentFactors.'])
elseif iVarOpt == nVarVals && nVarVals ~= 1
    warning(['The optimal geometric structure found occured at the ' ...
             'maximal supplied value for trial variance. Consider shifting ' ...
             'up the range of variances to be tested.'])
elseif any(optDisFactors==1) && ~any(find(optDisFactors==1) > nDichotToTune)
    warning(['The optimal geometric structure found occured at one of the ' ...
             'lower edges of the supplied search grid. Consider shifting ' ...
             'down the range of the displacmentFactors or increasing the ' ...
             'resolution of the search if feasible.'])
elseif iVarOpt == 1 && nVarVals ~= 1
    warning(['The optimal geometric structure found occured at the ' ...
             'minimal supplied value for trial variance. Consider shifting ' ...
             'down the range of variances to be tested or increasing the ' ...
             'resolution of the search, if feasible.'])
end

% Instantiate hypercube vertices using optimal parameters.
hypercuboid = embed_hypercuboid(nNeurons, nConds, optDisFactors, ...
                                'addTrials', false);

% Add gaussian noise to each vertex of the hypercube. This is done
% separately, because we might like to set 'tuneOnTrials' to false for
% speed, especially when there are many conditions/vertices.
fTxN = NaN(nTrials, nNeurons);
for iCond = 1:nConds  
    fTxN((iCond-1)*nTrialsPerCond + 1 : iCond*nTrialsPerCond, :) ...
        = hypercuboid(iCond,:) ...
          + randn(nTrialsPerCond, nNeurons) * sqrt(nvp.trialVar(iVarOpt));
end

% Calculate SD for factorizedTxN.
calculationTrialLabels = repelem((1:nConds)', nTrialsPerCond);
[factorized.sd, factorized.decoderPerf] ...
    = calc_sd(fTxN, calculationTrialLabels, ...
              'kFolds', nvp.kFolds, 'learner', ...
              'svm', 'pVal', false); 

% Calculate CCGP for full, noisy factorized model.
factorized.ccgp = test_ccgp(fTxN, calculationTrialLabels, ...
                            'learner', nvp.learner, 'pVal', nvp.pVal, ...
                            'nNull', nvp.nNull, 'nullMethod', nvp.nullMethod, ...
                            'returnNullDist', nvp.returnNullDist);  


end

function loss = calc_loss(y,yhat,metric)

assert(isvector(y) && isvector(yhat))
if isrow(y), y = y'; end
if isrow(yhat), yhat = yhat'; end

switch metric
    case 'sae' % sum of absolute erorrs
        loss = sum ( abs ( y - yhat) );
    case 'sse' % sum of squared errors
        loss = sum( (y - yhat).^ 2 );
    case 'rss' % root sum of squares = 2-norm of error vector
        loss = sqrt( sum( (y - yhat).^2 ) );
    case 'mse' % mean squared error
        loss = mean( (y - yhat).^2 ); 
end

end
