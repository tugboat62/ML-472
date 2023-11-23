import numpy as np
import cv2
from matplotlib import pyplot as plt

# Provide the path to the image file
image_path = "image.jpg"

# Read the image
image = cv2.imread(image_path)

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(image_rgb)
plt.axis('off')  # Turn off axis labels
plt.show()

##########################################################################

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the image
resized_image = cv2.resize(gray_image, (350, 500))
resized_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(image_rgb)
plt.axis('off')  # Turn off axis labels
plt.show()

# Display the grayscale image using matplotlib
plt.imshow(resized_rgb)
plt.axis('off')  # Turn off axis labels
plt.show()

# Convert the NumPy array to matrix A
A = np.array(gray_image)

# Print the shape of the matrix (height x width)
print("Shape of A:", A.shape)

# Print the matrix A
print("Matrix A:\n", A)

##############################################################################

# Perform Singular Value Decomposition
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# U, S, and Vt are the factorized matrices
# U and Vt are orthogonal matrices, and S is a 1D array of singular values

# Print the shapes of U, S, and Vt
print("Shape of U:", U.shape)
print("Shape of S:", S.shape)
print("Shape of Vt:", Vt.shape)

# Print the matrices U, S, and Vt
print("Matrix U:\n", U)
print("Matrix S:\n", np.diag(S))  # Display S as a diagonal matrix
print("Matrix Vt:\n", Vt)

##########################################################################

limit_k = min (A.shape)
print("Maximum size of k: ", limit_k)

##########################################################################

def low_rank_approximation(A, k):
    """
    Compute the k-rank approximation of matrix A using Singular Value Decomposition (SVD).

    Parameters:
    - A: Input matrix
    - k: Target rank for approximation

    Returns:
    - A_k: k-rank approximation of matrix A
    """

    # Truncate singular values and matrices to rank k
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    # Compute the k-rank approximation
    A_k = np.dot(U_k, np.dot(S_k, Vt_k))

    return A_k

##########################################################################

# By trial and error finding the lowest value of k such that the author's name can be read out clearly
# k's value is approximately 70. Can be lower if the person's eye-sight is super clear.
k = 70

approximation = low_rank_approximation(A, k)
# plot the matrix as grayscale image
plt.imshow(approximation, cmap='gray', interpolation='nearest')
plt.show()

print("Original Matrix A:\n", A)
print(f"\n{k}-Rank Approximation of A:\n", approximation)

##########################################################################

image_array = []

for i in range(1,4):
  k = i*10
  approximation = low_rank_approximation(A, k)
  image_array.append(approximation)

for i in range(1,4):
  k = i*50
  approximation = low_rank_approximation(A, k)
  image_array.append(approximation)

for i in range(1,5):
  k = 100 + i*200
  approximation = low_rank_approximation(A, k)
  image_array.append(approximation)

print(image_array.__len__())

##########################################################################

# Set the number of rows and columns in the grid
rows = 2
cols = 5

# Create a subplot grid
fig, axs = plt.subplots(rows, cols, figsize=(12, 6))

# Flatten the array of subplots for easier indexing
axs = axs.flatten()

# Plot each image
for i in range(len(image_array)):
    axs[i].imshow(image_array[i], cmap='gray')
    axs[i].axis('off')  # Turn off axis labels for clarity
    axs[i].set_title(f'Image {i+1}')

plt.tight_layout()
plt.show()
