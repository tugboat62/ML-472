import numpy as np

n = input("Enter the dimensions of the matrix: ")
n = int(n)

def generate_invertible_matrix(dim):
    while True:
        # Generate a random n x n matrix with integers
        matrix = np.random.randint(low=-10, high=10, size=(dim, dim))
        
        # Check if the determinant is non-zero
        if np.linalg.det(matrix) != 0:
            return matrix

# Example: Generate a random nxn invertible matrix
M = generate_invertible_matrix(n)
print(M)

eigen_values, eigen_vectors = np.linalg.eig(M)

print("Eigen values: ", eigen_values)
print("Eigen vectors: ", eigen_vectors)

P = eigen_vectors
if np.linalg.det(P) != 0 and eigen_values.all() != 0:
    print("Determinant of P is non-zero and all eigen values are non-zero.")
    P_inv = np.linalg.inv(P)
    A = P @ np.diag(eigen_values) @ P_inv    
    
    # Using np.allclose to check if two matrices are approximately equal
    if np.allclose(M, A):
        print("\nReconstruction Successful!")
        A = np.round(A).astype(int)
        print("\nReconstructed matrix: ")
        print(A)
        print("\nOriginal matrix: ")
        print(M)
    else:
        print("\nReconstruction Failed!")    

else:
    print("Determinant of P is zero or one or more eigen values are zero.")
    print("Reconstruction Failed!")
