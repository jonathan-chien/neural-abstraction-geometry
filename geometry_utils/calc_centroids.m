function CxN = calc_centroids(TxN,nConds)
% Calculate condition cluster centroids by averaging across all trials
% within each condition. 
%
% PARAMETERS
% ----------
% TxN    -- nTrials x nNeurons matrix of firing rates.
% nConds -- Scalar number of conditions (among the trials).
% 
% RETURNS
% -------
% CxN -- nConds x nNeurons of firing rates (corresponding to cluster
%        centroids).
%
% Author: Jonathan Chien 


% Determine number of neurons, trials, and trials per condition.
[nTrials, nNeurons] = size(TxN);
nTrialsPerCond = nTrials / nConds;
assert(mod(nTrials,nConds)==0, 'nTrials must be evenly divisible by nConds.')

% Average within condition for all conditions.
CxN = NaN(nConds, nNeurons);
for iCond = 1:nConds
    CxN(iCond,:) ...
        = mean(TxN((iCond-1)*nTrialsPerCond + 1 : iCond*nTrialsPerCond, :));
end

end