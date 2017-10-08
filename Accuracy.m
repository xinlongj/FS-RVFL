function [ErrorDis, Acc] = Accuracy(RealValue, EstimatedValue, Scale)

% Input:
% RealValue       - The real value calibrated by people
% EstimatedValue  - The estimated value, which is the output of learning models
% Scale           - The scale of pixel and distance
%
% Output:
% ErrorDis        - Error distance, here is 1 to 10 meters
% Acc             - Accuracy under different ErrorDis



ErrorDis = [1:1:10]';
Acc = zeros(10,1);

[m n] = size(RealValue);
Distance = zeros(m,1);
for i=1:1:m
    Distance(i) = norm(RealValue(i,:) - EstimatedValue(i,:));
end
Distance = Distance / Scale;

[p q] = size(ErrorDis);
for i=1:1:p
    for j=1:1:m
        if Distance(j) < ErrorDis(i)
            Acc(i) = Acc(i) + 1;
        end
    end
    Acc(i) = Acc(i) / m;
end
end