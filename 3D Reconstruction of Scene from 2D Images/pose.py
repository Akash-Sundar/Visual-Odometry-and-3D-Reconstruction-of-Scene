import numpy as np

def pose_candidates_from_E(E):
  transform_candidates = []
  ##Note: each candidate in the above list should be a dictionary with keys "T", "R"
  """ YOUR CODE HERE
  """

  U,S,V_T = np.linalg.svd(E)
  V = V_T.T

  R_pi_by_2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
  R_minus_pi_by_2 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
  #R_pi = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

  T = []
  T.append(U[:,-1])
  T.append(U[:,-1])
  T.append(-U[:,-1])
  T.append(-U[:,-1])

  R = []
  R.append(U @ R_pi_by_2.T @ V.T)
  R.append(U @ R_minus_pi_by_2.T @ V.T)
  R = R*2

  #transform_candidates.append([{'T': T[i]} for i in range(0,4)])
  #transform_candidates.append([{'R': R[i]} for i in range(0,4)])

  transform_candidates = [{'T':T[i], 'R': R[i]} for i in range(0,4)]


  """ END YOUR CODE
  """
  return transform_candidates