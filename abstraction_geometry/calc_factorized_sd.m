function factorized = calc_factorized_sd(sizes,targetCcgp,nvp)
% Calculates "shattering dimensionality" (SD) for a factorized geometry as
% described in Bernardi et al "The Geometry of Abstraction in the
% Hippocampus and Prefrontal Cortex," Cell, 2020.
%
% PARAMETERS
% ----------
% sizes      -- 3-vector whose elements specify the number of trials,
%               neurons, and conditions, respectively, to be used in
%               modeling the factorized model.
% targetCcgp -- Scalar value that is the target CCGP which we will try to
%               achieve by tuning the hypercube representing the factorized
%               model.
% Name-Value Pairs (nvp)
%   'displacementFactorVals' -- Vector of values that we will test over.
%                               value corresponds to one factorized model
%                               in which the length of the hypercube is
%                               scaled by that value.
%   'tuneWith'               -- String value, either 'accuracy' or 'AUC'.
%                               Specifies which performance metric should
%                               be compared against targetCcgp (this should
%                               correspond to the same performance metric
%                               as targetCcgp).
%   'kFolds'                 -- Once the length of the hypercube is tuned,
%                               guassian noise is added, and the shattering
%                               dimenisionality of this model is calculated
%                               via the calc_sd function. This name-value
%                               pair specifies the number of folds to use
%                               when cross validating the decoder
%                               performance on each of the dichtomies
%                               within that routine (briefly, SD is the
%                               average decoding performance over all
%                               dichotomies. See calc_sd documentation for
%                               more information).
%   'learner'                -- Once the length of the hypercube is tuned,
%                               guassian noise is added, and the shattering
%                               dimenisionality and CCGP of this model is
%                               calculated via the calc_sd function and
%                               test_ccgp function, respectively. For both
%                               of those routines, specify the linear
%                               classification model type with this
%                               name-value pair. Corresponds to 'Learner'
%                               name-value pair in MATLAB's fitclinear.
%   'pVal'                   -- Once the length of the hypercube is tuned,
%                               guassian noise is added, and the CCGP of
%                               this model is calculated via the test_ccgp
%                               function. For this routine, specify the p
%                               value type that should be attached to the
%                               CCGP values returned.
%   'nNull'                  -- Once the length of the hypercube is tuned,
%                               guassian noise is added, and the CCGP of
%                               this model is calculated via the test_ccgp
%                               function. For this routine, specify the
%                               number of null datasets used to generate
%                               the p values attached to the CCGP values
%                               returned.
%   'nullMethod'             -- Once the length of the hypercube is tuned,
%                               guassian noise is added, and the CCGP of
%                               this model is calculated via the test_ccgp
%                               function. For this routine, specify the
%                               method used to generate the null datasets.
%                               (see test_ccgp documentation for more
%                               information).
%                               
% RETURNS
% -------
% factorized -- 1 x 1 struct with the following fields:
%   .sd   -- 1 x 1 struct containing shattering dimensionality information
%            about the tuned factorized model (this is the output of the
%            calc_sd function when called on the factorized model).
%   .ccgp -- 1 x 1 struct containing CCGP information about the tuned
%            factorized model (this is the output of the test_ccgp function
%            when called on the factorized model).
%
% Author: Jonathan Chien Version 1.0. 7/24/21 Last edit: 8/24/21.


arguments
    sizes % a 3-vector = [nTrials nNeurons nConds]
    targetCcgp (1,1)
    nvp.displacementFactorVals = 0.1 : 0.1 : 2
    nvp.tuneWith string = 'accuracy' % accuracy or AUC
    nvp.kFolds = 5 % for testing sd of null model
    nvp.learner = 'svm' % for testing sd of null model
    nvp.pVal = false % This is for CCGP of null model
    nvp.nNull = 1000
    nvp.nullMethod = 'geometric'
end

% Determine number of trials, neurons, conditions, and trials per
% condition.
nTrials = sizes(1);
nNeurons = sizes(2);
nConds = sizes(3);
nTrialsPerCond = nTrials / nConds;
assert(mod(nTrials,nConds)==0, 'nTrials must be evenly divisible by nConds.')

% Check for potentially problematic values for nConds.
if nConds ~= 8 && nConds ~= 4
    warning(['This function currently supports mainly nConds == 4 or ' ...
             'nConds == 8. Behavior under other values for nConds is ' ...
             'not guaranteed.'])  
end

% Create trial labels and preallocate.
tuningTrialLabels = (1:nConds)'; % must column vector or test_ccgp will fail
ccgpDiff = NaN(length(nvp.displacementFactorVals), 1);

% Try different values for displacement, which tunes the length of the
% hypercube. 
parfor iDis = 1:length(nvp.displacementFactorVals)
    
    % Get current displacement factor.
    displacementFactor = nvp.displacementFactorVals(iDis);

    % Get vertices of cuboid embedded in N-space, with N = nNeurons.
    vertices = embed_hypercube(nNeurons, nConds, displacementFactor);
    
    % Get CCGP for current hypercube.
    ccgp = test_ccgp(vertices, tuningTrialLabels, 'pVal', false);
    switch nvp.tuneWith
        case 'accuracy'
            ccgpDiff(iDis) = abs(mean(ccgp.accuracy.ccgp) - targetCcgp);
        case 'AUC'
            ccgpDiff(iDis) = abs(mean(ccgp.auc.ccgp) - targetCcgp);
        otherwise
            error("Invalid value for 'tuneWith'.")
    end
end

% Select displacement factor leading to smallest error between target ccgp
% and average (across dichotomies) ccgp.
[~,iMin] = min(ccgpDiff);

% Construct hypercube with selected displacementFactor.
vertices = embed_hypercube(nNeurons, nConds, nvp.displacementFactorVals(iMin));

% Add guassian noise to each vertex of the hypercube. Perhaps gaussian
% noise clouds are relevant here as they may overlap for small lengths of
% hypercube.
nullTxN = NaN(nTrials, nNeurons);
for iCond = 1:nConds  
    % MxN is a point cloud around the ith vertex. Assign it into nullTxN.
    MxN = vertices(iCond,:) + randn(nTrialsPerCond, nNeurons);
    nullTxN((iCond-1)*nTrialsPerCond + 1 : iCond*nTrialsPerCond, :) = MxN;
end

% Calculate SD for nullTxN.
calculationTrialLabels = repelem(tuningTrialLabels, nTrialsPerCond);
[factorized.sd,~] = calc_sd(nullTxN, calculationTrialLabels, ...
                              'kFolds', nvp.kFolds, 'learner', 'svm', ...
                              'pVal', false);

% Calculate CCGP for full, noisy null model.
factorized.ccgp = test_ccgp(nullTxN, calculationTrialLabels, ...
                            'learner', nvp.learner, 'pVal', nvp.pVal, ...
                            'nNull', nvp.nNull, 'nullMethod', nvp.nullMethod);                     

end

function vertices = embed_hypercube(nNeurons,nConds,displacementFactor)
% Embed hypercube in n-space, with n = nNeurons. Currently, this works only
% with 4 conditions (in which case the hypercube is a square) and with 8
% conditions (in which case the hypercube is a cuboid, i.e. has 6 faces).

if nConds == 4
    % With n = nNeurons, begin with one random standard normal n-vector,
    % then perform the QR factorization of this vector to obtain Q, an n x
    % n unitary matrix. Take any two columns (or rows) of Q and their
    % respective reflections to obtain vertices of a square. Next calculate
    % the vector difference between two of these four vectors to obtain a
    % vector, d, parallel to two sides of the square. Multiply d by
    % displacementFactor, then apply + d to two vertices whose angle it
    % bisects, and -d to the other two vertices.
    [Q,~] = qr(randn(nNeurons, 1));
    vertices(1,:) = Q(:,1);
    vertices(2,:) = Q(:,2);
    vertices(3,:) = -vertices(1,:);
    vertices(4,:) = -vertices(2,:);
    d = (vertices(1,:) + vertices(2,:)) * displacementFactor; % v1 + v2 = v1 - (-v2) 
    vertices(1:2,:) = vertices(1:2,:) + d;
    vertices(3:4,:) = vertices(3:4,:) - d;
    
elseif nConds == 8
    % With n = nNeurons, begin with one random standard normal n-vector,
    % and perform QR decomposition to obtain Q, an n x n unitary matrix.
    % Select two columns (or rows) of Q and their reflections to define a
    % square. Then use any other column (or row) of Q (i.e., not among the
    % two already selected) as a displacement vector, d, to displace the
    % square twice, the first time scaling d by +1*displacementFactor and
    % the second time scaling d by -1*displacementFactor (in the opposite
    % direction). The absolute magnitude of displacementFactor thus
    % controls the length of the cuboid.
    [Q,~] = qr(randn(nNeurons, 1));
    vertices(1,:) = Q(:,1);
    vertices(2,:) = Q(:,2);
    vertices(3,:) = -vertices(1,:);
    vertices(4,:) = -vertices(2,:);
    vertices(1:4,:) = vertices(1:4,:) + displacementFactor * Q(3,:);
    vertices(5:8,:) = vertices(1:4,:) - displacementFactor * Q(3,:);
else
    error(['nConds was passed in as %d, which is not currently an ' ...
           'acceptable value.'], nConds)
end

end
