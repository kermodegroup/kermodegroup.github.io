import argparse
import numpy as np
from   tqdm  import tqdm

# Distance function
def dist(s_1, s_2):
    # Dot Product
    return 1 - s_2.dot(s_1)**2

    # Norm
#    return np.linalg.norm(s_1 - s_2, axis=1)
#======================================================================================

parser = argparse.ArgumentParser()
parser.add_argument("filename", metavar="f", type=str, help="The numpy binary file containing the SOAP feature vectors on each row of the matrix")
args = parser.parse_args()

#======================================================================================

# The 2d matrix containing the SOAP feature vectors on each row of the matrix
filename      = args.filename

# Load in the 2d matrix and transpose it so the feature vectors are the columns
data = np.load(filename).T

k    = []
rows = data.shape[0]
cols = data.shape[1]

# The number of features to select
# select all columns (can be manually overriden)
N    = cols

print("For the matrix ({},{}), we are selecting {} columns".format(rows,cols,N))

farthests    = np.zeros((N, rows))

# Randomly select the first column 
first        = np.random.randint(cols)

print("Selecting column:", first)
farthests[0] = data.T[first]

# Save the index of the first selected environment 
k.append(first)

# Make a 1D array for how far away each environment is from the first selected environment 
distances = dist(farthests[0], data.T)

# FPS loop
indexes = np.arange(0, N)
remaining_indexes = [True for i in indexes]

# Remove the first selected index
remaining_indexes[first] = False

for i in tqdm(indexes[1:]):
    # Find the index of the environment most far away
    pos = np.argmax(distances)
    
    # If we've already used this index we need to find the next largest unused one
    if pos in k:
        # The remaining unused indecies  
        indices = indexes[remaining_indexes]
        
        # build an array with distances on the top row and indexes on the bottom
        comb = np.vstack((distances[indices], indices))
        
        # Sort said array so that the largest distance is in the first column, with it's associated index        
        _sorted = np.flip(np.sort(comb), axis=1)
                
        # The index of the largest distance not yet used 
        pos = int(_sorted[1,0])
                           
    #--------------------------------------
    # Update the selected columns
    #--------------------------------------
    farthests[i] = data.T[pos]
    k.append(pos)
    
    # Set the mask so we know which values are unused
    remaining_indexes[pos] = False
    
    # Update the distances with any smaller distances from the new environment
    distances = np.minimum(distances, dist(farthests[i], data.T))


# Save
with open("{}.out".format(".".join(filename.split(".")[:-1])), "w") as f:
    f.writelines([str(l)+"\n" for l in k])
