import numpy as np
import matplotlib.pyplot as plt

def show_reprojections(image1, image2, uncalibrated_1, uncalibrated_2, P1, P2, K, T, R, plot=True):

  """ YOUR CODE HERE
  """

  P1proj = np.zeros([P1.shape[0], 3])
  P2proj = np.zeros([P1.shape[0], 3])

  for i in range(0, P1.shape[0]):
    P1proj[i,:] = R @ P1[i,:] + T
    P2proj[i,:] = R.T @ P2[i,:] - R.T @ T

    P1proj[i,:] = K @ P1proj[i,:]
    P2proj[i,:] = K @ P2proj[i,:]

  #P1proj = P1 @ R + T
  #P2proj = P2 @ R.T - R.T @ T

  #P1proj = (K @ P1proj.T).T
  #P2proj = (K @ P2proj.T).T
  
  """ END YOUR CODE
  """

  if (plot):
    plt.figure(figsize=(6.4*3, 4.8*3))
    ax = plt.subplot(1, 2, 1)
    ax.set_xlim([0, image1.shape[1]])
    ax.set_ylim([image1.shape[0], 0])
    plt.imshow(image1[:, :, ::-1])
    plt.plot(P2proj[:, 0] / P2proj[:, 2],
           P2proj[:, 1] / P2proj[:, 2], 'bs')
    plt.plot(uncalibrated_1[0, :], uncalibrated_1[1, :], 'ro')

    ax = plt.subplot(1, 2, 2)
    ax.set_xlim([0, image1.shape[1]])
    ax.set_ylim([image1.shape[0], 0])
    plt.imshow(image2[:, :, ::-1])
    plt.plot(P1proj[:, 0] / P1proj[:, 2],
           P1proj[:, 1] / P1proj[:, 2], 'bs')
    plt.plot(uncalibrated_2[0, :], uncalibrated_2[1, :], 'ro')
    
  else:
    return P1proj, P2proj