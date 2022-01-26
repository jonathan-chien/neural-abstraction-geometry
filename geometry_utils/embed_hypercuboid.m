function [hypercuboid,dichotomies] = embed_hypercuboid(nNeurons,nConds,disFactors,nvp)
% Embed hypercuboid in n-space, with n = nNeurons. Currently, this works
% only with 4 conditions (in which case the hypercube is a square) and with
% 8 conditions. disFactors is a vector whose length can range from
% 1 to 2 in the case of 4 conditions and 1 to 3 in the case of 8
% conditions. Each element of disFactors is a scalar controlling
% the amount by which one side of the hypercuboid will be stretched; if an
% element is zero, that respective side will not be stretched. If all
% elements are zero, the hypercuboid will have equal lengths on all sides.
% Returns vertices, an m x n matrix whose rows have n-vectors denoting the
% position of the m vertices in n-space. Also returns dichotomies array
% whose size is nDichot x 4 in the case of 4 conditions (where nDichot =
% length(disFactors), i.e. the number of sides of the hypercube to
% tune) and nDichot x 8 in the case of 8 conditions (where nDichot again =
% length(disFactors). Each row corresponds to the condition
% indices for one dichotomies, where the 1st and 2nd halves contain the
% indices for the conditions on each side of the dichotomy (this is
% analagous to the output of create_dichotomies.m; see that function's
% documentation for more information).

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
    % nDichotToTune. This can then be passed in to test_ccgp through the
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
    % nDichotToTune. This can then be passed in to test_ccgp through the
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
