function ifindex=Invert_file_index(dataset_dir,vocabulary_tree)
%Generate inverted file index

%List of images
vImgNames = dir(fullfile(dataset_dir, '*.png'));
nImgs = length(vImgNames);
assert(nImgs > 0);
fprintf('%d images\n', nImgs);

%Inverted file index: this is an array of size the number of leaves in the tree
%Each element of the array is a struct with two fields:
% - ifindex(i).images the images containing the visual word correponding to the leaf
% - ifindex(i).scores the score which the visual word will vote towards the corresponding image

%Initialize the inverted file index
nleaves=vocabulary_tree.K^vocabulary_tree.depth;
ifindex(nleaves).images=[];
ifindex(nleaves).scores=[];

% Extract features for all images  and fill in ifindex
%features_per_image=zeros(1,nImgs); %number of visual words in image i, used to normalize ifindex with tf-idf
%for i...
    %fprintf('Inverse file index: processing image %d, %s\n', i, vImgNames(i).name);

    % load the image
    %I=imread(fullfile(dataset_dir,vImgNames(i).name));

    %extract feature points with DoG and SIFT descriptors
    %...
    
    %Pass each descriptor down the vocabulary tree and accumulate
    %statistics for the respective image

    %for sdi = ... each descriptor in image i
            
        %...
        
        %path_to_leaf = ...
        
        %index=Path2index(path_to_leaf,vocabulary_tree.K);
        
        %Count occurrences of feature point sdi in image i:
        %increment score by 1 if image i is already in ifindex(index).images
        %add the new image (i) to ifindex(index).images if not present with score 1
        
    %features_per_image(i) = %

    
    
% Normalize ifindex with tf-idf
%ifindex=Norm_tf_idf(ifindex,features_per_image,nImgs);

