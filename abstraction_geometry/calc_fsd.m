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
%   'tuneOnTrials'   -- ('nTrialsPerCond'(default) | scalar | false)
%                       Specify the number of gaussian trials to be added
%                       to each vertex of the hypercube when tuning. If
%                       'nTrialsPerCond', the function will calculate the
%                       number of trials representing each condtiion in the
%                       actual data and use this value. Or, specifiy some
%                       nonnegative integer. Set to false to tune using
%                       classifiers trained on vertices of hypercube alone
%                       (not recommended for nConditions = 4, since
%                       classifiers will then train on two observations and
%                       usually have accuracy of 0 or 1 (and CCGP is the
%                       average over only 4 classifier's performances).
%                       Note well that this is is not related to the number
%                       of trials added to the vertices of the hypercube
%                       (after tuning) when calculating the SD and final,
%                       full CCGP (see 'nFinalTrials').
%   'trialVar'       -- (vector or scalar) Pass a range of variances of the
%                       gaussian distribution from which we make draws to
%                       construct a point cloud added to each vertex.
%                       (corresponding to simulated single trials). If
%                       'tuneOnTrials' ~= false, the function will test all
%                       variance values during the tuning step and select
%                       the optimal variance to apply to the point cloud
%                       surrounding vertices of the final, tuned hypercube
%                       during calculation of SD and the final, full CCGP.
%                       Note that if 'tuneOnTrials' = false, the first
%                       element of a vector argument for trialVar will be
%                       selected by default, but the function will still
%                       run a training loop over each value for 'trialVar'
%                       (so it may be in this case preferrable to instead
%                       pass in a scalar corresponding to the specific
%                       variance desired in the point cloud surrounding the
%                       vertices of the final, tuned hypercuboid to be
%                       tested; this avoids repeated and unnecessary
%                       testing in the tuning search).
%   'nFinalTrials'  -- ('nTrialsPerCond'(default) | scalar). Specify the
%                       number of trials to be added to the vertices of the
%                       final tuned hypercuboid (for calculation of CCGP,
%                       SD etc.), independently of the number of trials
%                       potentially used around each vertex during tuning.
%                       If 'nTrialsPerCond', the function will infer how
%                       many trials per condition are present in the actual
%                       data and use this value; otherwise, specify a
%                       nonnegative integer as the number of trials around
%                       each vertex.
%   'kFold'          -- (scalar) Once the length of the hypercube is tuned
%                       and gaussian noise added, the shattering
%                       dimenisionality of the factorized model is
%                       calculated via the calc_sd function. This
%                       name-value pair specifies the number of folds to
%                       use when cross validating the decoder performance
%                       on each of the dichotomies within that routine
%                       (briefly, SD is the average decoding performance
%                       over all dichotomies. See calc_sd documentation for
%                       more information).
%   'classifier'     -- (string) Once the length of the hypercube is tuned
%                       and gaussian noise added, the shattering
%                       dimenisionality and CCGP of the factorized model
%                       are calculated via the calc_sd function and
%                       calc_ccgp function, respectively. For both of those
%                       routines, specify the function to be used as a
%                       classifier. Note that due to the call to calc_sd,
%                       this function must have 'CrossVal' and 'KFold' as
%                       properties. Default is @fitclinear, which fits a
%                       linear SVM model.
%   'pVal'           -- ('left-tailed'|'right-tailed'|'two-tailed'|false
%                       (default)) Once the length of the hypercube is
%                       tuned and guassian noise added, the CCGP of this
%                       model is calculated via the calc_ccgp function. For
%                       this routine, pass a string specifying the type of
%                       test or set to logical false to suppress
%                       calculation of signficance for the CCGP of the
%                       dichotomies in the tuned factorized model.
%   'nNull'          -- (scalar: default = 1000) Once the length of the
%                       hypercube is tuned and gaussian noise added, the
%                       CCGP of this factorized model is calculated via the
%                       calc_ccgp function. For this routine, specify the
%                       number of null datasets used to generate the p
%                       values attached to the CCGP values returned.
%   'nullMethod'     -- (string: 'permutation'|'geometric') Once the length
%                       of the hypercube is tuned and gaussian noise added,
%                       the CCGP of this factorized model is calculated via
%                       the calc_ccgp function. For this routine, specify
%                       the method used to generate the null datasets. (see
%                       calc_ccgp documentation for more information).
%   'returnNullDist' -- (logical: 1|0 (default)). Specify whether or not
%                       to return the null distributions for accuracy and
%                       auc (for each dichotomy) in the ccgp field (see
%                       RETURNS below; also see the 'returnNullDist'
%                       Name-Value Pair in calc_ccgp for more information).
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
%                   calc_ccgp function when called on the factorized
%                   model).
%
% Author: Jonathan Chien 7/24/21. Last edit: 1/24/22.


arguments
    sizes 
    targetCcgp 
    nvp.disFactorVals = 1 : 0.1 : 2
    nvp.tuneWith string = 'accuracy' 
    nvp.lossFunction = 'sae'
    nvp.tuneOnTrials = 'nTrialsPerCond'
    nvp.trialVar = 0.025 
    nvp.nFinalTrials = 'nTrialsPerCond' 
    nvp.kFold = 5 
    nvp.classifier = @fitclinear 
    nvp.pVal = false 
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
assert(mod(nTrials,nConds)==0, 'nTrials must be evenly divisible by nConds.')
nDichotToTune = length(targetCcgp);
nDisFactorVals = length(nvp.disFactorVals);
nVarVals = length(nvp.trialVar);
if ~nvp.tuneOnTrials & nVarVals > 1
    warning("A range of trial variances was supplied, but tuning was " + ...
            "requested based on vertices only. The first variance value " + ...
            "will be used by default when constructing the final tuned " + ...
            "hypercube, so it is recommended to pass only the desired " + ...
            "point cloud variance as a scalar.")
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
    if strcmp(nvp.tuneOnTrials, 'nTrialsPerCond')
        nTuningTrials = nTrials / nConds;
    else
        assert(isscalar(nvp.tuneOnTrials) && mod(nvp.tuneOnTrials,1)==0, ...
               "If 'tuneOnTrials' is not 'nTrialsPerCond', it must be a " + ...
               "nonnegative integer.")
        nTuningTrials = nvp.tuneOnTrials;
    end
    tuningTrialLabels = repelem((1:nConds)', nTuningTrials);
else
    nTuningTrials = 0;
    tuningTrialLabels = (1:nConds)';
end

% Preallocate (same container size regardless of number of conditions due
% to 'omitnan' in min function).
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
                                        'addTrials', nTuningTrials, ...
                                        'trialVar', nvp.trialVar(iVar)); 
                
                % Calculate CCGPs of dichotomies for which the current
                % hypercube was tuned. For speed, CCGP is calculated only
                % for these specific dichotomies.
                ccgp = calc_ccgp(hypercuboid, tuningTrialLabels, ...
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
% speed, especially when there are many conditions/vertices. The number of
% trials added to the vertex can also be specified separately from the
% number used during tuning.
if strcmp(nvp.nFinalTrials, 'nTrialsPerCond')
    nFinalTrials = nTrials / nConds; 
else
    assert(nvp.nFinalTrials ~= 0)
    assert(isscalar(nvp.nFinalTrials) && mod(nvp.nFinalTrials,1)==0, ...
           "If 'nFinalTrials' is not 'nTrialsPerCond', it must be a " + ...
           "nonnegative integer.")
    nFinalTrials = nvp.nFinalTrials;
end
fTxN = NaN(nConds*nFinalTrials, nNeurons);
for iCond = 1:nConds  
    fTxN((iCond-1)*nFinalTrials + 1 : iCond*nFinalTrials, :) ...
        = hypercuboid(iCond,:) ...
          + randn(nFinalTrials, nNeurons) * sqrt(nvp.trialVar(iVarOpt));
end

% Calculate SD for factorizedTxN.
calculationTrialLabels = repelem((1:nConds)', nFinalTrials);
[factorized.sd, factorized.decoderPerf] ...
    = calc_sd(fTxN, calculationTrialLabels, ...
              'kFold', nvp.kFold, 'classifier', nvp.classifier, 'pVal', false); 

% Calculate CCGP for full, noisy factorized model.
factorized.ccgp = calc_ccgp(fTxN, calculationTrialLabels, ...
                            'classifier', nvp.classifier, 'pVal', nvp.pVal, ...
                            'nNull', nvp.nNull, 'nullMethod', nvp.nullMethod, ...
                            'returnNullDist', nvp.returnNullDist);  


end


% -------------------------------------------------------
function loss = calc_loss(y,yhat,metric)
% Calculate loss between two vector inputs.

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


% -------------------------------------------------------
function [hypercuboid,dichotomies] = embed_hypercuboid(nNeurons,nConds,disFactors,nvp)
% Tune and embed a hypercuboid in n-space, with n = nNeurons. Currently,
% this works only with 4 conditions (in which case the hypercuboid is a
% rectnangle) and with 8 conditions.
% 
% PARAMETERS
% ----------
% nNeurons   -- Scalar value equal to the number of neurons and to the
%               dimensionality of th embedding space.
% nConds     -- Scalar value equal to the number of condition centroids,
%               which are the vertices of the embedded hypercuboid.
% disFactors -- m-vector whose i_th element is a factor multiplying the
%               length of the displacement vector applied to the i_th
%               direction/side of the hypercube. If nConditions = 4, m <=
%               2. If nConditions = 8, m <= 3. If any element of m is zero,
%               the corresponding displacement vector will be 0 (that is,
%               that side will not be scaled).
% Name-Value Pairs (nvp)
%   'addTrials' -- (Scalar|false (default)). Specify the number of trials
%                  (drawn from a Gaussian distribution whose variance is
%                  controlled through 'trialVar' (see below)) to be placed
%                  around each vertex of the hypercuboid. For nConditions =
%                  4, there are only 4 decoders per dichotomy, each trained
%                  on only two points (if only vertices are used) so the
%                  addition of noise allows for smoother exploration of the
%                  loss landscape when tuning CCGP; also, the optimally
%                  tuned variance can be used when simulating trials in the
%                  final calculation of CCGP on the tuned hypercuboid. Set
%                  to false to return only the vertices of the hypercuboid.
%   'trialVar'  -- Scalar value specifying the variance of the Gaussian
%                  distribution from which simulated single trials are
%                  drawn to create point clouds around the condition
%                  centroids. If 'addTrials' is false, this value is
%                  ignored.
%
% RETURNS
% -------
% hypercuboid -- If 'addTrials' is a scalar, this is an
%                nConditions*nTrialsPerCondition x nNeurons array (where
%                nTrialsPerCondition = 'addTrials'); essentially this
%                simulates the neural population representation as 8
%                condition means situated at the vertices of a hypercuboid
%                in n-dimensional space (n = nNeurons), where each vertex
%                is surrounded by a point cloud simulating single trials of
%                that condition. If 'addTrials' = false, this is an
%                nConditions x nNeurons array, simulating the neural
%                population representation as 8 condition means only (i.e.,
%                without the simulated single trials).
% dichotomies -- m x nConditions array, where the i_th row contains
%                condition indices for the dichtomy whose separating
%                hyperplane is orthogonal to the side of the hypercuboid
%                tuned using the i_th element of disFactors. By returning
%                these indices, we can test only the dichotomies
%                corresponding to the scaled sides of the hypercuboid, thus
%                speeding up the grid search. Save for the variable number
%                of rows, this output is exactly the same as the first
%                output of create_dichotomies.
%
% Jonathan Chien. 1/20/22.

arguments
    nNeurons
    nConds
    disFactors
    nvp.addTrials = false % number of trials around each centroid
    nvp.trialVar = 1 % Variance of each point cloud, if trials added vertices
end

% Get number of displacement factors provided (the number of sides whose
% lengths we wish to tune).
nDichotToTune = length(disFactors);

if nConds == 4
    % With n = nNeurons, begin with one random standard normal n-vector,
    % then perform the QR factorization of this vector to obtain Q. Take
    % any two columns of Q and their respective reflections to obtain
    % vertices of a square (each side has length sqrt(2)). Next calculate
    % the vector difference between two orthogonal vectors out of these
    % four vectors to obtain a vector, d1, parallel to two sides of the
    % square; do the same with the other two vertices to get d2. Scale d1
    % and d2 by their respective factors (in the elements of
    % displacementFactor) and apply d1 and d2 to two respective sides of
    % the square.

    % Check number of displacement factors provided.
    assert(nDichotToTune <= 2)
    if nDichotToTune == 1, disFactors = [disFactors 0]; end

    % Set up hypercube (square).
    [Q,~] = qr(randn(nNeurons, 1));
    vertices(1,:) = Q(:,1);
    vertices(2,:) = Q(:,2);
    vertices(3,:) = -vertices(2,:);
    vertices(4,:) = -vertices(1,:);

    % Set up indices of two dichotomies that are tunable.
    dichotomies = [1:4; ...
                   [1 3 2 4]];

    % Calculate displacement vectors, multiply by displacement factors and
    % apply to vertices of hypercube. 
    d = NaN(2, nNeurons);
    d(1,:) = (vertices(1,:) - vertices(3,:)) * disFactors(1); 
    d(2,:) = (vertices(1,:) - vertices(2,:)) * disFactors(2); 
    for iDichot = 1:2
        vertices(dichotomies(iDichot,1:2),:) ...
            = vertices(dichotomies(iDichot,1:2),:) + d(iDichot,:);
        vertices(dichotomies(iDichot,3:4),:) ...
            = vertices(dichotomies(iDichot,3:4),:) - d(iDichot,:);
    end

    % Remove rows of "dichotomies" so that its 1st dim size is equal to
    % nDichotToTune. This can then be passed in to calc_ccgp through the
    % appropriate name-value pair.
    dichotomies = dichotomies(1:nDichotToTune,:);
    
elseif nConds == 8
    % With n = nNeurons, begin with one random standard normal n-vector,
    % and perform QR decomposition. Select two columns of Q and their
    % reflections to define a square. Then use any other column of Q (i.e.,
    % not among the two already selected) scaled by (sqrt(2)/2) as an
    % initial displacement vector, di, to displace the square twice, in
    % opposite directions (each orthogonal to the vector subspace in which
    % the square resides). The first displacment is by adding di, and the
    % second is by adding -di. Next calculate d1, d2, and d3 in a manner
    % analagous to the 4 condition case above and apply them to the
    % vertices of the hypercube, in order to tune the length of its sides.

    % Check number of displacement factors provided.
    assert(nDichotToTune <= 3)
    if nDichotToTune < 3
        disFactors = [disFactors zeros(1, 3-nDichotToTune)];
    end

    % Set up hypercube (3D). Displace vertices 5-8 first.
    [Q,~] = qr(randn(nNeurons, 1));
    vertices(1,:) = Q(:,1);
    vertices(2,:) = Q(:,2);
    vertices(3,:) = -vertices(2,:);
    vertices(4,:) = -vertices(1,:);
    vertices(5:8,:) = vertices(1:4,:) - (sqrt(2)/2) * Q(:,3)';
    vertices(1:4,:) = vertices(1:4,:) + (sqrt(2)/2) * Q(:,3)'; 
    
    % Set up indices of the 3 dichotomies that could be tuned. 
    dichotomies = [1:8; ...
                   [1 2 5 6 3 4 7 8]; ...
                   [1 3 5 7 2 4 6 8]];

    % Calculate displacement vectors (d1,d2,d3), scale them by their
    % respective factors, and apply them to vertices of hypercube.
    d = NaN(3, nNeurons);
    d(1,:) = (vertices(1,:) - vertices(5,:)) * disFactors(1);
    d(2,:) = (vertices(1,:) - vertices(3,:)) * disFactors(2);
    d(3,:) = (vertices(1,:) - vertices(2,:)) * disFactors(3);
    for iDichot = 1:3
        vertices(dichotomies(iDichot,1:4),:) ...
            = vertices(dichotomies(iDichot,1:4),:) + d(iDichot,:);
        vertices(dichotomies(iDichot,5:8),:) ...
            = vertices(dichotomies(iDichot,5:8),:) - d(iDichot,:);
    end

    % Remove rows of "dichotomies" so that its 1st dim size is equal to
    % nDichotToTune. This can then be passed in to calc_ccgp through the
    % appropriate name-value pair.
    dichotomies = dichotomies(1:nDichotToTune,:);

else
    error(['nConds was passed in as %d, which is not currently an ' ...
           'acceptable value.'], nConds)
end

% Option to simulate point cloud of trials around each vertex with draws
% from a normal distribution (variance = 'trialVar').
if nvp.addTrials
    hypercuboid = NaN(nConds * nvp.addTrials, nNeurons);
    for iCond = 1:nConds
        hypercuboid((iCond-1)*nvp.addTrials+1 : iCond*nvp.addTrials, :) ...
            = vertices(iCond,:) ...
              + randn(nvp.addTrials, nNeurons) * sqrt(nvp.trialVar);
    end
else
    hypercuboid = vertices;
end

end
