function factorized = calc_fsd(TxN,condLabels,targetCcgp,nvp)
% Calculates "shattering dimensionality" (SD) for a factorized geometry as
% described in Bernardi et al "The Geometry of Abstraction in the
% Hippocampus and Prefrontal Cortex," Cell, 2020. Briefly, the i_th point
% cloud consisting of single trial population vectors from the i_th
% condition, centered at the i_th condition mean, is placed at the
% corresponding i_th vertex of a hypercuboid. Trials for each neuron may
% then be resampled within condition. Next, the lengths of the hypercube
% are tuned such that the CCGP calculated on these data match the target
% CCGP from the empirical data. Finally, the shattering dimensionality is
% calculated on this "factorized" model; when compared with the empirical
% shattering dimensionality, this yields some measure of how far beyond
% linear combinations of the task variables the neural encoding goes.
%
% PARAMETERS
% ----------
% TxN        -- m x n array of neural firing rates, where m = nTrials, and
%               n = nNeurons. Note that all conditions are represented
%               among the m trials, but it is not required that they be
%               represented in equal number.
% condLabels -- Vector of length m (where m = nTrials), where the i_th
%               element is the condition index of the i_th single trial.
% targetCcgp -- Vector whose elements contain the target CCGPs which we
%               will try to achieve by tuning the hypercuboid representing
%               the factorized model. For 8 conditions, this vector may
%               have 1 - 3 elements; for 4 conditions, it may have 1 or 2
%               elements.
% Name-Value Pairs (nvp)
%   'disFactorVals' -- 2D array of values to be tested over when
%                      tuning lengths of the hypercuboid. Each row
%                      corresponds to one dichotomy and to a set of factors
%                      for tuning one side of the hypercuboid (the
%                      direction orthogonal to a separating hyperplane for
%                      that dichotomy). The maximum number of
%                      rows/decodable dichotomies is 2 for the 4 condition
%                      case and 3 for the 8 condition case (for m
%                      conditions as the vertices of an m-dimensional
%                      hypercuboid, the instrinsic dimensionality of the
%                      hypercuboid (and thus the dimension of the embedding
%                      vector space if the hypercuboid is centered at the
%                      origin) is log2(m) = n; in such an n-dimensional
%                      space, there are n translational degrees of freedom,
%                      and linear separation requires us to be able to move
%                      from one class to another in a single direction;
%                      hence n orthogonal linear separations are possible).
%                      If a vector is passed in, but the number of target
%                      CCGP values is greater than one, the vector of
%                      values will be applied to each of the sides of the
%                      hypercuboid to be tuned. Note as well that the
%                      number of models tested is the number of tuning
%                      values raised to the power of the number of
%                      dichotomies, so overly fine-grained searches may
%                      come with a very high computational cost, especially
%                      in the 8 condition case.
%   'tuneWith'      -- (string: 'accuracy'| 'auc') Specify which
%                      classification performance metric should be compared
%                      against targetCcgp (this should correspond to the
%                      same performance metric as targetCcgp).
%   'lossFunction'  -- (string: 'sae'|'sse'|'rss'|'mse') Specify
%                      the function used to calculate error between a tuned
%                      hypercuboid's CCGPs and the passed in target CCGPs.
%   'calcCcgp'      -- (1 (default) | 0), specify whether or not to
%                      calculate the CCGP of the final, tuned, factorized
%                      model (this is not related to tuning). If so,
%                      parameters for this routine are set via the
%                      'ccgpParams' name-value pair (but see the note about
%                      the classifier field of 'ccgpParams' below). This
%                      can be set to false to suppress the final CCGP
%                      calculation if, for example, we wish to conduct a
%                      rough-grained pilot search, to be followed by a
%                      subsequent fine-grained search around the optimal
%                      parameters of the rough search.
%   'ccgpParams'    -- Scalar struct. Once the length of the hypercuboid is
%                      tuned and gaussian noise added, the CCGP of the
%                      factorized model can be calculated via the calc_ccgp
%                      function. For this routine, specify the name-value
%                      pairs for the calc_ccgp function in the fields of a
%                      scalar struct passed in through 'ccgpParams'. The
%                      passed in values will be checked, and if any
%                      parameters are not specified, default values will be
%                      specified (see local function parse_ccgp_inputs;
%                      note that these default settings will override the
%                      default name-value pair values in the calc_ccgp
%                      function). NB: the value of ccgpParams.classifier
%                      will be used to specify the classifier used both to
%                      calculate the final CCGP and during the tuning step.
%                      Unless otherwise specified, it is set as
%                      @fticlinear; this value is used even if 'calcCcgp'
%                      is false. All other fields of ccgpParams are unused
%                      if 'calcCcgp' is false.
%   'calcSd'        -- (1 (default) | 0), specify whether or not to
%                      calculate the SD of the final, tuned, factorized
%                      model. If so, parameters for this routine are set
%                      via the 'sdParams' name-value pair (see the
%                      'decoderParams' name-value pair of the calc_sd.m
%                      function for more information). This can be set to
%                      false to suppress the final SD calculation if for
%                      example, we wish to conduct a rough-grained pilot
%                      search, to be followed by a subsequent fine-grained
%                      search around the optimal parameters of the rough
%                      search.
%   'sdParams'      -- Scalar struct. Once the length of the hypercube is
%                      tuned and gaussian noise added, the shattering
%                      dimenisionality of the factorized model can be
%                      calculated via the calc_sd function. This name-value
%                      pair passes in decoding parameters to calc_sd. See
%                      calc_sd documentation for more information). If
%                      passed in as empty (as it is by default), default
%                      parameters will be set inside calc_sd. If 'calcSd' =
%                      false, this argument is ignored.
%                               
% RETURNS
% -------
% factorized -- Scalar struct with the following fields:
%   .fTxN          -- nTrials x nNeurons matrix of firing rates for the
%                     optimal factorized model (used to calculate final
%                     SD and CCGP values etc).
%   .sd            -- Scalar struct containing shattering dimensionality
%                     information about the tuned factorized model (this is
%                     the output of the calc_sd function when called on the
%                     factorized model). This field is only returned if
%                     'calcSd' = true (see Name-Value Pairs under
%                     PARAMETERS).
%   .decoderPerf   -- Scalar struct containing information about decoding
%                     performance (accuracy and auc) for each dichotomy of
%                     the factorized model (see performance under RETURNS
%                     in the documentation for calc_sd or performance under
%                     RETURNS in the documentation for multiset_decoder.m
%                     for more information). This field is only returned if
%                     'calcSd' = true (see Name-Value Pairs under
%                     PARAMETERS).
%   .decoderParams -- Scalar struct whose fields contain the parsed params
%                     that were passed to multiset_decoder.m (the parsing
%                     was performed in the calc_sd function, which wraps
%                     multiset_decoder and is wrapped by calc_fsd_2.m).
%                     This field is only returned if 'calcSd' = true (see
%                     Name-Value Pairs under PARAMETERS).
%   .ccgp          -- Scalar struct containing CCGP information about the
%                     tuned factorized model (this is the output of the
%                     calc_ccgp function when called on the factorized
%                     model). This field is only returned if 'calcCcgp' =
%                     true (see Name-Value Pairs under PARAMETERS).
%   .ccgpParams    -- Scalar struct containing the parsed params that were
%                     passed to the calc_ccgp function during the final
%                     testing of CCGP of the tuned, factorized model. Note
%                     that the classifier specified here is also what is
%                     used during the tuning step. This field is returned
%                     even if 'calcCcgp' = false (see Name-Value Pairs
%                     under PARAMETERS), but its only field is .classifier.
%   .optimization  -- Scalar struct containing information related to
%                     tuning of the geometric structure.
%       .optDisFactors -- Vector whose elements are the optimal
%                         displacement factors selected via tuning and used
%                         to construct a hypercuboid with CCGP of the
%                         target dichotomies most closely matching that
%                         observed on the empirical data.
%       .ccgpLoss      -- 3D array whose i_th, j_th, k_th element is the
%                         loss of the tuned vs target CCGP for the i_th,
%                         j_th, k_th of the three respective dichotomies.
%                         Note that if fewer than 3 target dichtomies were
%                         supplied, their respective dimensions in this
%                         array will have NaN values.
%       .minLoss       -- Minimum value among all non-NaN elements of
%                         ccgpLoss.
%
% Author: Jonathan Chien. 3/18/22. Last updated 3/23/22.


arguments
    TxN
    condLabels
    targetCcgp 
    nvp.disFactorVals = -0.8 : 0.1 : 1.0
    nvp.tuneWith string = 'accuracy' 
    nvp.oversampledTuning = 100
    nvp.lossFunction string = 'sae'
    nvp.calcCcgp = true
    nvp.ccgpParams = [];
    nvp.calcSd = true
    nvp.sdParams = []
end


%% Parse and check arguments

% Determine number of trials, conditions, and neurons.
nNeurons = size(TxN, 2);
condInd = unique(condLabels);
nConds = length(condInd);
nDichotToTune = length(targetCcgp);
nDisFactorVals = length(nvp.disFactorVals);

% Separate point clouds for each condition and mean-center each cloud.
condClouds = cell(nConds, 1);
condCloudLabels = cell(nConds, 1);
condCloudsOversampled = cell(nConds, 1);
condCloudLabelsOversampled = cell(nConds, 1);
for iCond = 1:nConds
    % Get point cloud and labels for current condition.
    condClouds{iCond} = TxN(condLabels==condInd(iCond),:);
    condCloudLabels{iCond} = condLabels(condLabels==condInd(iCond));
    
    % Option to oversample within current condition point cloud.
    if nvp.oversampledTuning
        oversampled = NaN(nvp.oversampledTuning, nNeurons);
        for iNeuron = 1:nNeurons
            oversampled(:,iNeuron) ...
                = datasample(condClouds{iCond}(:,iNeuron), ...
                             nvp.oversampledTuning, 'Replace', true);
        end

        condCloudsOversampled{iCond} = oversampled;
        condCloudLabelsOversampled{iCond} ...
            = repelem(condInd(iCond), nvp.oversampledTuning)';

    % If not oversampling, condCloudsOversampled will be equivalent to
    % condClouds (same for the labels). 
    else
        condCloudsOversampled{iCond} = condClouds{iCond};
        condCloudLabelsOversampled{iCond} = condCloudLabels{iCond};
    end

    % Mean-center point cloud and oversampled point cloud.
    condClouds{iCond} = condClouds{iCond} - mean(condClouds{iCond}, 1);  
    condCloudsOversampled{iCond} ...
        = condCloudsOversampled{iCond} - mean(condCloudsOversampled{iCond}); 
end

condCloudLabels = vertcat(condCloudLabels{:});
condCloudLabelsOversampled = vertcat(condCloudLabelsOversampled{:});

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

% If disFactorVals passed in as vector but nDichotToTune > 1, duplicate
% vector for each dichotomy.
if isvector(nvp.disFactorVals)
    expandedDisFactorVals = repmat(nvp.disFactorVals, nDichotToTune, 1);
elseif ismatrix(nvp.disFactorVals)
    expandedDisFactorVals = nvp.disFactorVals;
    assert(size(expandedDisFactorVals, 1) == nDichotToTune, ...
           "If passing in a matrix of tuning factors, the number of rows" + ...
           "must match the number of target CCGPs supplied.");
end

% Expand the tuning factors into an array whose first dim size matches the
% number of dichotomies we would like to tune. Then pad with rows of zeros
% so that the first dim size = 3.
if nDichotToTune < 3
    expandedDisFactorVals = [expandedDisFactorVals; ...
                             zeros(3-nDichotToTune, nDisFactorVals)];
end



%% Tune hypercuboid

% Parse calc_ccgp parameters (passed in through the 'ccgpParams' name-value
% pair) here, since calc_ccgp is called during the tuning step, and the
% classifier must be set. 
ccgpParamsParsed = parse_ccgp_params(nvp.ccgpParams);

% Preallocate (same container size regardless of number of conditions due
% to 'omitnan' in min function).
ccgpLoss = NaN(nDisFactorVals, nDisFactorVals, nDisFactorVals);

% Try all combinations of different values for displacement, which tunes
% the length of the hypercube. 
parfor iFactor1 = 1:nDisFactorVals
    for iFactor2 = 1:nDisFactorVals
        for iFactor3 = 1:nDisFactorVals            
            
            % Obtain current set of displacement scaling factors.
            currDisFactors = [expandedDisFactorVals(1,iFactor1) ...
                              expandedDisFactorVals(2,iFactor2) ...
                              expandedDisFactorVals(3,iFactor3)];
            if nConds == 4, currDisFactors(3) = []; end
    
            % Using current displacement factors, embed hypercuboid in
            % N-space (N = nNeurons) and optionally add empirical single
            % trials around each condition's mean, situated at a vertex of
            % the hypercuboid.
            [fTxNOversampled, dichotomies] ...
                = create_factorized(nNeurons, nConds, currDisFactors, ...
                                    condCloudsOversampled); 
            
            % Calculate CCGPs of dichotomies for which the current
            % hypercube was tuned. For speed, CCGP is calculated only
            % for these specific dichotomies.
            ccgp = calc_ccgp(fTxNOversampled, condCloudLabelsOversampled, ...
                             'classifier', ccgpParamsParsed.classifier, ...
                             'dichotomies', dichotomies, 'pval', false);
            
            % Calculate loss using the specified performance metric and
            % loss function.
            [ccgpLoss(iFactor1,iFactor2,iFactor3), ~] ...
                        = calc_loss(targetCcgp, ...
                                    ccgp.(nvp.tuneWith).ccgp(1:nDichotToTune), ...
                                    nvp.lossFunction);
                 
            % Breaking condition to prevent unnecessary repetitions.
            if nConds == 4 || nDichotToTune < 3, break; end
        end
    
        % Breaking condition to prevent unnecessary repetitions.
        if nDichotToTune == 1, break; end
    end
end


%% Construct optimal tuned hypercuboid and calculate CCGP and SD

% Select displacement factors leading to smallest error between target
% CCGPs and constructed CCGP(s). 
[minLoss, iMin] = min(ccgpLoss, [], 'all', 'omitnan', 'linear');
[iFactor1Opt,iFactor2Opt,iFactor3Opt] ...
    = ind2sub(repmat(nDisFactorVals, 1, 3), iMin);
optDisFactors = [expandedDisFactorVals(1,iFactor1Opt) ...
                 expandedDisFactorVals(2,iFactor2Opt) ...
                 expandedDisFactorVals(3,iFactor3Opt)];
if nConds == 4, optDisFactors(3) = []; end

% If one of the optimal parameters was the largest or smallest provided,
% warn user that search grid may need to be shifted/modified to better
% cover loss basin.
if any(optDisFactors == nDisFactorVals) 
    warning(['The optimal geometric structure found occured at one of the ' ...
             'upper edges of the supplied search grid. Consider shifting ' ...
             'up the range of the displacmentFactors.'])
elseif any(optDisFactors==1) && ~any(find(optDisFactors==1) > nDichotToTune)
    warning(['The optimal geometric structure found occured at one of the ' ...
             'lower edges of the supplied search grid. Consider shifting ' ...
             'down the range of the displacmentFactors or increasing the ' ...
             'resolution of the search if feasible.'])
end

% Instantiate factorized model using optimal parameters.
fTxN = create_factorized(nNeurons, nConds, optDisFactors, condClouds);
fTxNOversampled = create_factorized(nNeurons, nConds, optDisFactors, condCloudsOversampled);
factorized.fTxN = fTxN; % Save this model (no oversampling) to be returned

% Calculate SD for factorized model. Here we use the original pseudotrials
% (no oversampling).
if nvp.calcSd
    [factorized.sd, factorized.decoderPerf, factorized.decoderParams] ...
        = calc_sd(fTxN, condCloudLabels, 'decoderParams', nvp.sdParams);
end

% Calculate CCGP for factorized model.
if nvp.calcCcgp
    factorized.ccgp ...
        = calc_ccgp(fTxNOversampled, condCloudLabelsOversampled, ...
                    'classifier', ccgpParamsParsed.classifier, ...
                    'pval', ccgpParamsParsed.pval, ...
                    'nullInt', ccgpParamsParsed.nullInt, ...
                    'nNull', ccgpParamsParsed.nNull, ...
                    'nullMethod', ccgpParamsParsed.nullMethod, ...
                    'permute', ccgpParamsParsed.permute, ...
                    'returnNullDist', ccgpParamsParsed.returnNullDist);  
    factorized.ccgpParams = ccgpParamsParsed;
else
    factorized.ccgpParams = ccgpParamsParsed;

    % Remove field names that are not classifier. 
    fnames = fieldnames(factorized.ccgpParams);
    for iField = 1:length(fnames)
        if ~strcmp(fnames{iField}, 'classifier')
            factorized.ccgpParams = rmfield(factorized.ccgpParams, fnames{iField});
        end
    end
end


% Store information related to tuning, including optimal displacement
% factors, loss, and minimum loss.
factorized.optimization.optDisFactors = optDisFactors;
factorized.optimization.ccgpLoss = ccgpLoss;
factorized.optimization.minLoss = minLoss;


end


function parsedParams = parse_ccgp_params(passedParams)
% Each field in the defaultParams struct is checked against the fields of
% the passed in params (passedParams). If a match is found, the value of
% the field in passedParams is assigned to the field of the same name in
% parsedParams. If no match is found, the value of the field from
% defaultParams is used instead. Note that any extraneous fields in
% passedParams will thus be ignored. 

if isempty(passedParams), passedParams = struct('aaa', []); end

% Create struct with default params.
defaultParams = struct('condNames', [], 'dropInd', [], 'dichotomies', [], ...
                       'classifier', @ficlinear, 'pval', 'two-tailed', ...
                       'nullInt', 95, 'nNull', 1000, ...
                       'nullMethod', 'permutation', 'permute', 'neurons', ...
                       'returnNullDist', false);

% Create a struct with empty fields whose names match those in default
% params.
fnamesDefault = fieldnames(defaultParams);
for iField = 1:length(fnamesDefault)
    parsedParams.(fnamesDefault{iField}) = [];
end

% Get all field names in passed params struct.
fnamesPassed = fieldnames(passedParams);

% For each field in parsedParams, use the passed in value if passedParams
% has a field of the same name; else use the value from defaultParams.
for iField = 1:length(fnamesDefault)
    field = fnamesDefault{iField};

    if cell2mat(strfind(fnamesPassed, field))
        parsedParams.(field) = passedParams.(field);
    else
        parsedParams.(field) = defaultParams.(field);
    end
end

end


% -------------------------------------------------------
function [loss,vecDiff] = calc_loss(y,yhat,metric)
% Calculate loss between two vector inputs.

assert(isvector(y) && isvector(yhat))
if isrow(y), y = y'; end
if isrow(yhat), yhat = yhat'; end

vecDiff = y - yhat;

switch metric
    case 'sae' % sum of absolute erorrs
        loss = sum ( abs (vecDiff) );
    case 'sse' % sum of squared errors
        loss = sum( (vecDiff).^ 2 );
    case 'rss' % root sum of squares = 2-norm of error vector
        loss = sqrt( sum( (vecDiff).^2 ) );
    case 'mse' % mean squared error
        loss = mean( (vecDiff).^2 ); 
end

end


% -------------------------------------------------------
function [fTxN,dichotomies] = create_factorized(nNeurons,nConds,disFactors,condClouds)
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
% condClouds -- nConds x 1 or 1 x nConds cell array where the i_th cell
%               contains an m x n matrix, where m = the number of empirical
%               single trials from the i_th condition and n = the number of
%               neurons in the population. 
%
% RETURNS
% -------
% fTxN        -- nTrials x nNeurons array (where all conditions are
%                represented (though not necessarily equally) among the
%                nTrials ); essentially this simulates the neural
%                population representation as 8 condition means situated at
%                the vertices of a hypercuboid in n-dimensional space (n =
%                nNeurons), where each vertex is surrounded by the recorded
%                (not simulated) point cloud of single trials of the
%                corresponding condition (some effort was made to ensure
%                that the vertex indices/dichotomies match up with the
%                conditions/dichotomies we are working with).
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

% Specify whether or not to rotate each condition's point cloud.
ROTATE = true;

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

    % Calculate displacement vectors (d1, d2, d3), scale them by their
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

% Add point clouds to corresponding vertices. Option to rotate point cloud
% by permuting neuron labels, controlled by ROTATE.
fTxN = [];
for iCond = 1:nConds
    if ROTATE, condClouds{iCond} = condClouds{iCond}(:,randperm(nNeurons)); end
    fTxN = [fTxN; vertices(iCond,:) + condClouds{iCond}];
end

end
