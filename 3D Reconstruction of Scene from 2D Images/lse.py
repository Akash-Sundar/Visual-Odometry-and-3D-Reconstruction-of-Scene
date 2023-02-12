import numpy as np

def least_squares_estimation(X1, X2):
  """ YOUR CODE HERE
  """
  p = np.zeros([3,1])
  q = np.zeros([3,1])

  a = []

  for i in range(0, X1.shape[0]):
    p = X1[i].T
    q = X2[i].T

    a.append(np.hstack([p[0]*q.T, p[1]*q.T, p[2]*q.T]))

  a = np.array(a)
  #print(a.shape)

  U,S,V_T = np.linalg.svd(a)
  V = V_T.T
  E = V[:,-1]

  #print(E.shape)
  E = np.reshape(E,[3,3]).T

  U,S,V_T = np.linalg.svd(E)
  M = np.eye(3)
  M[-1,-1] = 0
  #print(M)
  E = U @ M @ V_T



  """ END YOUR CODE
  """
  return E
