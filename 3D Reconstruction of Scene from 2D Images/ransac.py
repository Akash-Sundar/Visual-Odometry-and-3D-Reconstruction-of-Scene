from lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8

    eps = 10**-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None

    #num_iterations = 100

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]

        """ YOUR CODE HERE
        """
        X1_train = X1[sample_indices]
        X2_train = X2[sample_indices]

        e3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        
        test_inliers = []
        
        E = least_squares_estimation(X1_train, X2_train)

        for i in test_indices:
            d1 = (X2[i].T @ E @ X1[i])**2/(np.linalg.norm(e3 @ E @X1[i]))**2
            d2 = (X1[i].T @ E.T @ X2[i])**2/(np.linalg.norm(e3 @ E.T @ X2[i]))**2
            d = d1+ d2
            if d < eps:
              test_inliers.append(i)
        inliers = np.concatenate((sample_indices, np.array(test_inliers)), axis=None)

        """ END YOUR CODE
        """
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = inliers

    #print(best_E.shape)
    #print(best_inliers.shape)
    return best_E, best_inliers