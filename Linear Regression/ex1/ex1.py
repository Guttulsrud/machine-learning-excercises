# Modified to fit Python, instead of Octave/MATLAB

# The first part of ex1.m gives you practice with Octave/MATLAB syntax and the homework submission process. In the
# file warmUpExercise.m, you will find the outline of an Octave/MATLAB function. Modify it to return a 5 x 5 identity
# matrix by filling in the following code: A = eye(5); 1Octave is a free alternative to MATLAB. For the programming
# exercises, you are free to use either Octave or MATLAB. 2 When you are finished, run ex1.m (assuming you are in the
# correct directory, type “ex1” at the Octave/MATLAB prompt) and you should see output similar to the following: ans
# = Diagonal Matrix 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 Now ex1.m will pause until you press any key,
# and then will run the code for the next part of the assignment. If you wish to quit, typing ctrl-c will stop the
# program in the middle of its run

import numpy as np

print(np.identity(5))


