import cv2
import numpy as np
import matplotlib.pyplot as plt

def projectionImage(H, img1):
  pt1 = H @ np.array([0, 0, 1])
  pt2 = H @ np.array([img1.shape[1], 0, 1])
  pt3 = H @ np.array([0, img1.shape[0], 1])
  pt4 = H @ np.array([img1.shape[1], img1.shape[0], 1])
  
  min_x = int(min(pt1[0]/pt1[2], pt2[0]/pt2[2], pt3[0]/pt3[2], pt4[0]/pt4[2]))
  max_x = int(max(pt1[0]/pt1[2], pt2[0]/pt2[2], pt3[0]/pt3[2], pt4[0]/pt4[2]))
  min_y = int(min(pt1[1]/pt1[2], pt2[1]/pt2[2], pt3[1]/pt3[2], pt4[1]/pt4[2]))
  max_y = int(max(pt1[1]/pt1[2], pt2[1]/pt2[2], pt3[1]/pt3[2], pt4[1]/pt4[2]))

  img1_topleft_coords = np.array([min_x, min_y])
  img1_warped = np.zeros((max_y-min_y, max_x-min_x, 3))

  for i in range(img1.shape[0]):
      for j in range(img1.shape[1]):
          coords = H @ np.array([j, i, 1])
          x = int(coords[0]/coords[2] - img1_topleft_coords[0])
          y = int(coords[1]/coords[2] - img1_topleft_coords[1])
          
          if y < img1_warped.shape[0] and x < img1_warped.shape[1]:
              img1_warped[y][x] = img1[i][j]

        
  plt.figure()
  plt.imshow(img1_warped/255)
  inv_H = np.linalg.inv(H)

  for i in range(img1_warped.shape[0]):
      for j in range(img1_warped.shape[1]):
          if img1_warped[i][j].all() == 0:
              
              y = int(j + img1_topleft_coords[0])
              x = int(i + img1_topleft_coords[1])
              
              coords = inv_H @ np.array([y, x, 1])
              x = int(coords[0]/coords[2])
              y = int(coords[1]/coords[2])
              if y >= 0 and y <= 757 and x >= 0 and x <= 567:
                  img1_warped[i][j] = img1[y][x] 
                  
  plt.figure()
  plt.imshow(img1_warped/255)
  
  img1_warped = img1_warped.astype(np.uint8)

  return img1_warped, img1_topleft_coords

def mergeWarpedImages(img1_warped, img2, img1_topleft_coords):  
  stitched_image = np.zeros((int(img2.shape[0] + abs(img1_topleft_coords[1])), int(img2.shape[1] + abs(img1_topleft_coords[0])), 3))

  for i in range(stitched_image.shape[0]):
      for j in range(stitched_image.shape[1]):
          x = int(i + img1_topleft_coords[1])
          y = int(j + img1_topleft_coords[0])
          
          if (x < 0 and j < img1_warped.shape[1]) or (y < 0 and i < img1_warped.shape[0]):
             stitched_image[i][j] = img1_warped[i][j]
              
          elif x >= 0 and x < img2.shape[0] and y >= 0 and y < img2.shape[1] :            
             stitched_image[i][j] = img2[x][y]
              
  stitched_image = stitched_image.astype(np.uint8)
  
  return stitched_image


def stitchImages(img1, img2):
  SIFT = cv2.xfeatures2d.SIFT_create()
  query_features, descriptors1 = SIFT.detectAndCompute(img1, None)
  train_features, descriptors2 = SIFT.detectAndCompute(img2, None)
  drawimg1 = cv2.drawKeypoints(img1, query_features, None)
  drawimg2 = cv2.drawKeypoints(img2, train_features, None)
  plt.figure()
  plt.imshow(drawimg1)
  plt.figure()
  plt.imshow(drawimg2)
  FLANN_INDEX_KDTREE = 0
  flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  flann = cv2.FlannBasedMatcher(flann_params, {})
  matches = flann.knnMatch(descriptors1, descriptors2, k=2)
  filtered_matches = []
  for match in matches:
      if match[0].distance/match[1].distance < 0.8:
          filtered_matches.append(match[0])

  matching = cv2.drawMatches(img1, query_features, img2, train_features, filtered_matches, None)
  plt.figure()
  plt.imshow(matching)   
  query = np.array([query_features[match.queryIdx].pt for match in filtered_matches]).reshape(-1, 1, 2).astype(np.float32)
  
  train = np.array([train_features[match.trainIdx].pt for match in filtered_matches]).reshape(-1, 1, 2).astype(np.float32)

  H, _ = cv2.findHomography(query, train, cv2.RANSAC, 5.0)

  img1_warped, img1_topleft_coords = projectionImage(H, img1)

  stitched_image = mergeWarpedImages(img1_warped, img2, img1_topleft_coords)

  return stitched_image

images = []
for i in range(6):
    image = cv2.imread("cv22_lab2_material/part3 - ImageStitching/{}.jpg".format(i+1))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)


images = np.array(images)

img1 = images[0]
img2 = images[1]
img3 = images[2]
img = stitchImages(img1, img2)
plt.figure()
plt.imshow(img)

upper_img = stitchImages(img3, img)

plt.figure()
plt.imshow(upper_img)

img1 = images[3]
img2 = images[4]
img3 = images[5]
img = stitchImages(img1, img2)
plt.figure()
plt.imshow(img)

lower_img = stitchImages(img3, img)

plt.figure()
plt.imshow(lower_img)


panorama = stitchImages(lower_img, upper_img)

plt.figure()
plt.imshow(panorama)

