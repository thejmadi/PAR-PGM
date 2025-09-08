% Munkres Test

tracks = [1,1; 2,2; 3,3];
dets = [3.1, 3.1; 0.9, 0.9; 1.8, 2.2];

for i = size(tracks, 1):-1:1
    delta = dets - tracks(i, :);
    costMatrix(i, :) = sqrt(sum(delta .^ 2, 2));
end
costofnonassignment = 0.2;

[assignments, unassignedTracks, unassignedDetections] = ...
    assignmunkres(costMatrix,costofnonassignment);
disp(assignments);
disp(unassignedDetections);
