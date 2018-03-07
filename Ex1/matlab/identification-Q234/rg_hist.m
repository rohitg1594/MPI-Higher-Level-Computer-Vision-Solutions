%
%  compute joint histogram for r/g values
%  note that r/g values should be in the range [0, 1];
%  histogram should be normalized so that sum of all values equals 1
%
%  img_color - input color image
%  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
%

function h = rg_hist(img_color, num_bins)

  assert(size(img_color, 3) == 3, 'image dimension mismatch');
  assert(isfloat(img_color), 'incorrect image type');
  
  %define a 2D histogram  with "num_bins^2" number of entries
  h=zeros(num_bins, num_bins);

  % ...

