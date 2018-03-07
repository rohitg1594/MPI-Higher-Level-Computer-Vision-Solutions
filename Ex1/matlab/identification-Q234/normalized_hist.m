%
%  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
%  assume that image intensity varies between 0 and 255
%
%  img_gray - input image in grayscale format
%  num_bins - number of bins in the histogram
%
function h = normalized_hist(img_gray, num_bins)

  assert(length(size(img_gray)) == 2, 'image dimension mismatch');
  assert(isfloat(img_gray), 'incorrect image type');

  % compute the histogram of pixel intensity values
  % ...

  %normalize the histogram such that its integral (sum) is equal 1
  % ...