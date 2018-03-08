function number_of_leaves=Countleaves(parent)
%Function to count the vocabulary words, i.e. the number of
%cluster centroids at the leaves of the vocabulary tree.

%A possible implementation is recursive:

%IF at a leaf node
%   count the centroids
%   ...
%ELSE
%   keep traversing the tree (call Countleaves iteratively) to the leaf nodes
%   ...
