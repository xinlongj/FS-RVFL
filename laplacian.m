
function L = laplacian(DATA, TYPE, options)  

% Calculate the graph laplacian of the adjacency graph of data set DATA.
%
% L = laplacian(DATA, TYPE, PARAM)  
% 
% DATA - NxK matrix. Data points are rows. 
% TYPE - string 'nn' or string 'epsballs'
% options - Data structure containing the following fields
% NN - integer if TYPE='nn' (number of nearest neighbors), 
%       or size of 'epsballs'
% 
% DISTANCEFUNCTION - distance function used to make the graph
% WEIGHTTYPPE='binary' | 'distance' | 'heat'
% WEIGHTPARAM= width for heat kernel
% NORMALIZE= 0 | 1 whether to return normalized graph laplacian or not 
%
% Returns: L, sparse symmetric NxN matrix 
%
% Author: 
%
% Mikhail Belkin 
% misha@math.uchicago.edu
%
% Modified by: Vikas Sindhwani (vikass@cs.uchicago.edu)
% June 2004

%joker disp('Computing Graph Laplacian.');


NN=options.NN; 
DISTANCEFUNCTION=options.GraphDistanceFunction;
WEIGHTTYPE=options.GraphWeights;
WEIGHTPARAM=options.GraphWeightParam;
NORMALIZE=options.GraphNormalize;




% calculate the adjacency matrix for DATA
A = adjacency(DATA, TYPE, NN, DISTANCEFUNCTION);
  
W = A;

% disassemble the sparse matrix
[A_i, A_j, A_v] = find(A);

switch WEIGHTTYPE
    
case 'distance'
   for i = 1: size(A_i)  
       W(A_i(i), A_j(i)) = A_v(i);
   end;
  
case 'binary'
%joker  disp('Laplacian : Using Binary weights ');
    for i = 1: size(A_i)  
       W(A_i(i), A_j(i)) = 1;
    end;
 
case 'heat' 
%joker     disp(['Laplacian : Using Heat Kernel sigma : ' num2str(WEIGHTPARAM)]);
    t=WEIGHTPARAM;
    for i = 1: size(A_i)  
       W(A_i(i), A_j(i)) = exp(-A_v(i)^2/(2*t*t));
    end;
    
otherwise
    error('Unknown Weighttype');   
end

D = sum(W(:,:),2);   

if NORMALIZE==0
    L = spdiags(D,0,speye(size(W,1)))-W;
else % normalized laplacian
    D=diag(sqrt(1./D));
    L=eye(size(W,1))-D*W*D;
    L(find(isnan(L)==1)) = 0 ; % joker copy from ljf
end
