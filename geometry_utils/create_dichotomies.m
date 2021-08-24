function [dichotomies,dichotomyConds] = create_dichotomies(nConds,condLabels)
% Accepts as input nConds, the number of conditions, and an optional
% argument condLabels, a 1 x nConds cell array containing the names of each
% of the nConds conditions. Returns dichotomies, a nDichotomies x nConds
% matrix, where the first 1:m column elements of the i_th row (with m =
% nConds/2) are the indices of conditions on one side of the i_th dichotomy
% and elements m+1:end of the i_th row are the indices of the conditions on
% the other side of the i_th dichotomy. dichotomyConds is a cell array of
% the same size as dichotomies, but each element contains a string array
% which is the name of a condition, rather than its index.
%
% PARAMETERS
% ----------
% nConds     -- Scalar value that is the number of conditions.
% condLabels -- 1 x nConds cell array, where each cell contains a string
%               which is the name of a condition (and that condition's
%               index is the same as the index of its name in condLabels).
%               This argument is optional, and the user may also supply an
%               empty array, in which case dichotomyConds will be returned
%               as an empty array as well.
%
% RETURNS
% -------
% dichotomies    -- nDichotomies x nConds matrix. The i_th row corresponds
%                   to the i_th dichotomy, with the first half of the i_th
%                   row elements being the indices of conditions on one
%                   side of the dichotomy, and the latter half the indices
%                   of the conditions on the other side of the dichotomy.
%                   nDichotomies is calculated from nConds as detailed in
%                   the inline comments in the function below.
% dichotomyConds -- nDichotomies x nConds cell array. Each cell has a
%                   corresponding element in dichotomies. However, whereas
%                   the elements of dichotomies contain condition indices,
%                   the cells of dichotomyConds contain condition names. If
%                   condLabels was empty, this variable will also be
%                   returned as an empty array.
%                   
% Author: Jonathan Chien 7/20/21 Version 1.0 Last edit: 7/20/21.
% Based on methods section from Bernardi et al "The Geometry of Abstraction
% in the Hippocampus and Prefrontal Cortex," Cell, 2020.


% Set combination parameter m, where we want to partition m*n objects into
% n unique groups, with n = 2 (hence a "dichotomy").
m = nConds/2;
   
% Get dichotomies and store corresponding condition labels in cell array
% dichotomyConds.
dichotomies = [nchoosek(1:nConds,m) flip(nchoosek(1:nConds,m),1)];
nDichotomies = size(dichotomies, 1)/2;
dichotomies = dichotomies(1:nDichotomies,:);
if ~isempty(condLabels)
    dichotomyConds(:,1:m) = condLabels(dichotomies(:,1:m));
    dichotomyConds(:,m+1:nConds) = condLabels(dichotomies(:,m+1:nConds));
else
    dichotomyConds = [];
end

end