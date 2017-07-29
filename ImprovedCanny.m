function [e] = ImprovedCanny(a,method)
[a,sigma,k,filter] = parse_inputs(a,method);

a=double(a);
original=a;
[m,n] = size(a);
%output edges
e = false(m,n);

a= smoothGradient(a, sigma,filter);
dx=zeros(m,n);
dy=zeros(m,n);
 %calculate gradient vectors along x direction and y direction
for i=2:m-1
    for j=2:n-1        
         dx(i, j)=(a(i+1,j)-a(i-1,j)) + (sqrt(2)/4) * (a(i+1,j-1)-a(i-1,j+1)+a(i+1,j+1)-a(i-1,j-1));         
         dy(i, j)=(a(i,j+1)-a(i,j-1)) + (sqrt(2)/4) * (a(i-1,j+1)-a(i+1,j-1)+a(i+1,j+1)-a(i-1,j-1));                 
    end;
end;
    
% Calculate Magnitude of Gradient
magGrad = hypot(dx, dy);
    
% Normalize for threshold selection
magmax = max(magGrad(:));
if magmax > 0
    magGrad = magGrad / magmax;
end

if(strcmp(method,'rich'))
       e= richEdge(e,dx, dy, magGrad,k);
else
       e= weakEdge(e,dx,dy,magGrad,k);        
end
imtool(e,[]);
 
%calculate snr 
ima=max(e(:));
imi=min(e(:));
ims=std(e(:));
ImprovedCanny_snr=20*log10((ima-imi)./ims);
     
im_canny=edge(original,'canny');
imtool(im_canny,[]);
ima=max(original(:));
imi=min(original(:));
ims=std(original(:));
Canny_snr=20*log10((ima-imi)./ims);
         
function idxLocalMax = cannyFindLocalMaxima(direction,ix,iy,mag)

[m,n] = size(mag);

% Find the indices of all points whose gradient (specified by the
% vector (ix,iy)) is going in the direction we're looking at.

switch direction
    case 1
        idx = find((iy<=0 & ix>-iy)  | (iy>=0 & ix<-iy));
    case 2
        idx = find((ix>0 & -iy>=ix)  | (ix<0 & -iy<=ix));
    case 3
        idx = find((ix<=0 & ix>iy) | (ix>=0 & ix<iy));
    case 4
        idx = find((iy<0 & ix<=iy) | (iy>0 & ix>=iy));
end

% Exclude the exterior pixels
if ~isempty(idx)
    v = mod(idx,m);
    extIdx = (v==1 | v==0 | idx<=m | (idx>(n-1)*m));
    idx(extIdx) = [];
end

ixv = ix(idx);
iyv = iy(idx);
gradmag = mag(idx);

% Do the linear interpolations for the interior pixels
switch direction
    case 1
        d = abs(iyv./ixv);
        gradmag1 = mag(idx+m).*(1-d) + mag(idx+m-1).*d;
        gradmag2 = mag(idx-m).*(1-d) + mag(idx-m+1).*d;
    case 2
        d = abs(ixv./iyv);
        gradmag1 = mag(idx-1).*(1-d) + mag(idx+m-1).*d;
        gradmag2 = mag(idx+1).*(1-d) + mag(idx-m+1).*d;
    case 3
        d = abs(ixv./iyv);
        gradmag1 = mag(idx-1).*(1-d) + mag(idx-m-1).*d;
        gradmag2 = mag(idx+1).*(1-d) + mag(idx+m+1).*d;
    case 4
        d = abs(iyv./ixv);
        gradmag1 = mag(idx-m).*(1-d) + mag(idx-m-1).*d;
        gradmag2 = mag(idx+m).*(1-d) + mag(idx+m+1).*d;
end
idxLocalMax = idx(gradmag>=gradmag1 & gradmag>=gradmag2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Local Function : parse_inputs
%
function [a,Sigma,k,filter] = parse_inputs(a,Method)
% OUTPUTS:
%   I      Image Data
%   Method Edge detection method
%   Thresh Threshold value
%   Sigma  standard deviation of Gaussian
%   H      Filter for Zero-crossing detection
%   kx,ky  From Directionality vector
if(strcmp(Method,'rich'))
    Sigma=2;
    k=1.2;
    filter=16;
elseif(strcmp(Method,'weak'))
    Sigma=1.4;
    k=1.6;
    filter=16;
else
    error(message('Method should be rich or weak'));
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Local Function : smoothGradient
%
function [a] = smoothGradient(I, sigma,filterLength)

% Create an even-length 1-D separable Derivative of Gaussian filter

% Determine filter length

n = (filterLength - 1)/2;
x = -n:n;
y=-n:n;

% Create 1-D Gaussian Kernel
c = 1/(2*pi*sigma*sigma);
gaussKernel = c * exp(-(x.^2+y.^2)/(2*sigma^2));

% Normalize to ensure kernel sums to one
gaussKernel = gaussKernel/sum(gaussKernel);
a = imfilter(I, gaussKernel, 'conv', 'replicate');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Local Function : weakEdge
%
function e = weakEdge(e,dx, dy, magGrad,k)

%Find the mean and standard deviation of the gradiant magnitude

avg = mean2(magGrad);
std=std2(magGrad);
highThresh = avg + (k*std);
lowThresh = highThresh/ 2;
%For suppresion and checking with thresholds
e = thinAndThresholdWeak(e, dx, dy, magGrad, lowThresh, highThresh);

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Local Function : richEdge
%
function e = richEdge(e,dx, dy, magGrad,k)

%Find threshold for every pixel by finding mean of the neighborhod window
avg = mean2(magGrad);
kernel = ones(35)/35^35; % Create averaging window.
avg_mat = conv2(magGrad, kernel, 'same');
std_mat = stdfilt(magGrad, ones(35));
highThresh_mat = avg_mat + (k .* std_mat);
lowThresh_mat = highThresh_mat ./ 2;  
e = thinAndThresholdRich(e,dx, dy, magGrad, lowThresh_mat, highThresh_mat,avg);
  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Local Function : thinAndThresholdRich
%
function H = thinAndThresholdRich(E, dx, dy, magGrad, lowThresh, highThresh,avg)

% Perform Non-Maximum Suppression Thining and Hysteresis Thresholding of Edge
% Strength
    
% We will accrue indices which specify ON pixels in strong edgemap
% The array e will become the weak edge map.
idxStrong = [];
for dir = 1:4
    idxLocalMax = cannyFindLocalMaxima(dir,dx,dy,magGrad);
    idxLocalMax=idxLocalMax-(magGrad(idxLocalMax) < 0.2*avg);
    idxWeak = idxLocalMax -(magGrad(idxLocalMax) > lowThresh(idxLocalMax));
    E(idxWeak)=1;
    idxStrong = [idxStrong; idxWeak(magGrad(idxWeak) > highThresh(idxWeak))]; %#ok<AGROW>
end

[m,n] = size(E);

if ~isempty(idxStrong) % result is all zeros if idxStrong is empty
    rstrong = rem(idxStrong-1, m)+1;
    cstrong = floor((idxStrong-1)/m)+1;
    H = bwselect(E, cstrong, rstrong, 8);
else
    H = zeros(m, n);
end
