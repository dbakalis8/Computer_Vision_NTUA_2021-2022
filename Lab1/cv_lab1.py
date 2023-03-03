import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Circle
import cv22_lab1_part3_utils as p3

def LoG(x, y, s):
    nom = y**2 + x**2 - 2*s**2 
    denom = 2*np.pi*(s**6)
    expo = np.exp(-(x**2 + y**2)/(2*s**2))
    return nom*expo/denom

def disk_strel(n):
    r = int(np.round(n))
    d = 2*r+1
    x = np.arange(d) - r
    y = np.arange(d) - r
    x, y = np.meshgrid(x,y)
    strel = x**2 + y**2 <= r**2
    return strel.astype(np.uint8)

def interest_points_visualization(I_, kp_data_, ax=None):
    I = np.array(I_)
    kp_data = np.array(kp_data_)
    assert(len(I.shape) == 2 or (len(I.shape) == 3 and I.shape[2] == 3))
    assert(len(kp_data.shape) == 2 and kp_data.shape[1] == 3)

    if ax is None:
        _, ax = plt.subplots()

    ax.set_aspect('equal')
    ax.imshow(I, 'gray')
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    for i in range(len(kp_data)):
        x, y, sigma = kp_data[i]
        circ = Circle((x, y), 3*sigma, edgecolor='g', fill=False, linewidth=1)
        ax.add_patch(circ)

    return ax

def EdgeDetect(img, s = 2, theta = 0.1, linear = False):
    n = int(np.ceil(3*s)*2 + 1)
    B = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ], dtype=np.uint8)
    G1D = cv2.getGaussianKernel(n, s)
    G2D = G1D @ G1D.T
    smoothed_img = cv2.filter2D(img, -1, G2D)
    
    if linear == True:
        kernel = np.zeros((n,n))
        x = y = np.linspace(-(n-1)/2, (n-1)/2, n)
        x, y = np.meshgrid(x, y)
        kernel = LoG(x, y, s)
    
# =============================================================================
#         print (kernel)
#         plt.figure()
#         ax = plt.axes(projection='3d')
#         ax.plot_surface(x, y, kernel)
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax.set_zlabel('z')
# =============================================================================
    
        laplacian = cv2.filter2D(img, -1, kernel)
        
# =============================================================================
#         plt.figure()
#         plt.imshow(laplacian, 'gray')
# =============================================================================
        
    else:
        dilated_img  = cv2.dilate(smoothed_img, B)
        eroded_img = cv2.erode(smoothed_img, B)
        
        laplacian = dilated_img + eroded_img - 2*smoothed_img
        
# =============================================================================
#         plt.figure()
#         plt.imshow(laplacian, 'gray')
# =============================================================================
        
    _, binary_img = cv2.threshold(laplacian, 0, 1, cv2.THRESH_BINARY)
    y = cv2.dilate(binary_img, B) - cv2.erode(binary_img, B)
    ix, iy = np.gradient(smoothed_img)
    ig = np.sqrt(ix**2 + iy**2)
    max_g = ig.max()
    y_final = np.zeros((y.shape[0], y.shape[1]))
    
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i][j] == 1 and ig[i][j] > theta*max_g:
                y_final[i][j] = 1

    return y_final

def CornerDetect(img, s = 2, r = 2.5, k = 0.05, theta = 0.005):
    n = int(np.ceil(3*s)*2 + 1)
    
    G1D = cv2.getGaussianKernel(n, s)
    G2D = G1D @ G1D.T
    r1D = cv2.getGaussianKernel(n, r)
    r2D = r1D @ r1D.T
    
    smoothed_img = cv2.filter2D(img, -1, G2D)
    ix, iy = np.gradient(smoothed_img)
    
    j1 = cv2.filter2D(ix*ix, -1, r2D)
    j2 = cv2.filter2D(ix*iy, -1, r2D)
    j3 = cv2.filter2D(iy*iy, -1, r2D)

    l1 = (j1 + j3 + np.sqrt((j1 - j3)*(j1 - j3) + 4*j2*j2))/2
    l2 = (j1 + j3 - np.sqrt((j1 - j3)*(j1 - j3) + 4*j2*j2))/2
    
# =============================================================================
#     plt.figure()
#     plt.imshow(l1, 'gray')
#     plt.figure()
#     plt.imshow(l2, 'gray')
# =============================================================================
    
    R = l1*l2 - k*(l1 + l2)*(l1 + l2)
    B_sq = disk_strel(n)
    Cond1 = (R==cv2.dilate(R,B_sq))
    Cond2 = (R > theta*R.max())

    R = Cond1*Cond2
    
    x = []
    y = []
    
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i][j]:
                x.append(int(j))
                y.append(int(i))
                
    S = len(x)*[s]
    interest_points = np.array([x, y, S]).T
    
    return interest_points

def MultiscaleCornerDetect(img, s1 = 2, r = 2.5, k = 0.05, theta = 0.005, s2 = 1.5, N = 4):
    LoG = []
    i_points = []
    s = []
    for i in range(N):
        s.append(s1*s2**i)
        r = r*s2**i
        
        interest_points = CornerDetect(img, s[i], r, k, theta)
        i_points.append(interest_points)
        
        n = int(np.ceil(3*s[i])*2 + 1)
        
        G1D = cv2.getGaussianKernel(n, s[i])
        G2D = G1D @ G1D.T
        
        smoothed_img = cv2.filter2D(img, -1, G2D)
        ix, iy = np.gradient(smoothed_img)
        ixx, _ = np.gradient(ix)
        _, iyy = np.gradient(iy)
        LoG.append(s[i]**2*np.abs(ixx + iyy))
        
    interest_points = []
    for i, scale_points in enumerate(i_points):
        for points in scale_points:
            x = int(points[0])
            y = int(points[1])
            s = points[2]
            if i == 0 and LoG[i][y][x] > LoG[i+1][y][x]:
                interest_points.append([x, y, s])
            elif i == N-1 and LoG[i][y][x] > LoG[i-1][y][x]:
                interest_points.append([x, y, s])
            elif LoG[i][y][x] > LoG[i-1][y][x] and LoG[i][y][x] > LoG[i+1][y][x]:
                interest_points.append([x, y, s])
    return np.array(interest_points)
        
def BlobDetect(img, s = 2, theta = 0.005):
    n = int(np.ceil(3*s)*2 + 1)
    
    G1D = cv2.getGaussianKernel(n, s)
    G2D = G1D @ G1D.T
    
    smoothed_img = cv2.filter2D(img, -1, G2D)
    ix, iy = np.gradient(smoothed_img)
    ixx, ixy = np.gradient(ix)
    _, iyy = np.gradient(iy)
    
    R = ixx*iyy - ixy*ixy
    
    B_sq = disk_strel(n)
    Cond1 = (R==cv2.dilate(R,B_sq))
    Cond2 = (R > theta*R.max())
    
    R = Cond1*Cond2
    
    x = []
    y = []
    
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i][j]:
                x.append(int(j))
                y.append(int(i))
                
    S = len(x)*[s]
    interest_points = np.array([x, y, S]).T
    
    return interest_points

def MultiscaleBlobDetect(img, s1 = 2, theta = 0.005, s2 = 1.5, N = 4):
    LoG = []
    i_points = []
    s = []
    for i in range(N):
        s.append(s1*s2**i)
        
        blobs = BlobDetect(img, s[i], theta)
        i_points.append(blobs)
        
        n = int(np.ceil(3*s[i])*2 + 1)
        
        G1D = cv2.getGaussianKernel(n, s[i])
        G2D = G1D @ G1D.T
        
        smoothed_img = cv2.filter2D(img, -1, G2D)
        ix, iy = np.gradient(smoothed_img)
        ixx, _ = np.gradient(ix)
        _, iyy = np.gradient(iy)
        
        LoG.append(s[i]**2*np.abs(ixx + iyy))
        
    blobs = []
    for i, scale_points in enumerate(i_points):
        for points in scale_points:
            x = int(points[0])
            y = int(points[1])
            s = points[2]
            if i == 0 and LoG[i][y][x] > LoG[i+1][y][x]:
                blobs.append([x, y, s])
            elif i == N-1 and LoG[i][y][x] > LoG[i-1][y][x]:
                blobs.append([x, y, s])
            elif LoG[i][y][x] > LoG[i-1][y][x] and LoG[i][y][x] > LoG[i+1][y][x]:
                blobs.append([x, y, s])
    return np.array(blobs)

def BoxFilter(int_img, s):
    n = int(np.ceil(3*s)*2 + 1)
    
    h = 4*np.floor(n/6) + 1
    
    if h%3 != 0:
        h = h + 3 - h%3
        if (h/3) % 2 == 0:
            h += 3
        
    w = 2*np.floor(n/6) + 1
    
    padded_int_img = np.pad(int_img, (int((h-1)/2), int((h-1)/2))) 
        
    x = padded_int_img - np.roll(padded_int_img, int((h-1)/2-1), axis = 0) - np.roll(padded_int_img, int((w-1)/2 - 1), axis = 1) + np.roll(padded_int_img, (int((h-1)/2-1), int((w-1)/2 - 1)), axis = (0, 1))
    lxx = x + np.roll(x, int(2*h/3), axis=0) - 2*np.roll(x, int(h/3), axis=0)

    y = padded_int_img - np.roll(padded_int_img, int((h-1)/2 - 1), axis = 1) - np.roll(padded_int_img, int((w-1)/2 - 1), axis = 0) + np.roll(padded_int_img, (int((w-1)/2 - 1), int((h-1)/2 - 1)), axis = (0, 1))
    lyy = y + np.roll(y, int(2*h/3), axis=1) - 2*np.roll(y, int(h/3), axis=1)
    
    xy = padded_int_img - np.roll(padded_int_img, int((w-1)/2 - 1), axis = 0) - np.roll(padded_int_img, int((w-1)/2 - 1), axis = 1) + np.roll(padded_int_img, (int((w-1)/2 - 1), int((w-1)/2 - 1)), axis = (0, 1))
    lxy =  xy + np.roll(xy, (int((w-1)/2 - 1), int((w-1)/2 - 1)), axis = (0, 1))  - np.roll(xy, int((w-1)/2 - 1), axis = 0) - np.roll(xy, int((w-1)/2 - 1), axis = 1)      
    
    return lxx, lyy, lxy

def BlobDetectWithBoxFilters(img, s = 2, theta = 0.005):
    n = int(np.ceil(3*s)*2 + 1)
    G1D = cv2.getGaussianKernel(n, s)
    G2D = G1D @ G1D.T
    
    smoothed_img = cv2.filter2D(img, -1, G2D)
    
    int_img = np.cumsum(np.cumsum(smoothed_img, axis = 1), axis = 0)
    
    lxx, lyy, lxy = BoxFilter(int_img, s)
    ix, iy = np.gradient(smoothed_img)
    
    f = 15
    lxx = lxx[f:-f, f:-f]
    lyy = lyy[f:-f, f:-f]
    lxy = lxy[f:-f, f:-f]
    
    R = lxx*lyy - 0.81*lxy*lxy
    
    padx = img.shape[0] - R.shape[0]
    pady = img.shape[1] - R.shape[1]
    
    B_sq = disk_strel(n)
    Cond1 = (R ==cv2.dilate(R ,B_sq))
    Cond2 = (R > theta*R.max())
    
    R = Cond1*Cond2
    
    if padx>0 and pady>0:
        R = np.pad(R, ((padx//2 , padx//2), (pady//2, pady//2)))
    else:
        R = R[padx//2:-padx//2, pady//2:-pady//2]
    
    x = []
    y = []
    
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i][j]:
                x.append(int(j))
                y.append(int(i))
                
    S = len(x)*[s]
    interest_points = np.array([x, y, S]).T
    
    return interest_points

def MultiscaleBlobDetectWithBoxFilters(img, s1 = 1.7, theta = 0.005, s2 = 1.3, N = 4):
    LoG = []
    i_points = []
    s = []
    for i in range(N):
        s.append(s1*s2**i)
        
        blobs = BlobDetectWithBoxFilters(img, s[i], theta)
        i_points.append(blobs)
        
        n = int(np.ceil(3*s[i])*2 + 1)
        G1D = cv2.getGaussianKernel(n, s[i])
        G2D = G1D @ G1D.T
        smoothed_img = cv2.filter2D(img, -1, G2D)
        
        ix, iy = np.gradient(smoothed_img)
        ixx, _ = np.gradient(ix)
        _, iyy = np.gradient(iy)
        LoG.append(s[i]**2*np.abs(ixx + iyy))
        
    blobs = []
    for i, scale_points in enumerate(i_points):
        for points in scale_points:
            x = int(points[0])
            y = int(points[1])
            s = points[2]
            if i == 0 and LoG[i][y][x] > LoG[i+1][y][x]:
                blobs.append([x, y, s])
            elif i == N-1 and LoG[i][y][x] > LoG[i-1][y][x]:
                blobs.append([x, y, s])
            elif LoG[i][y][x] > LoG[i-1][y][x] and LoG[i][y][x] > LoG[i+1][y][x]:
                blobs.append([x, y, s])
                
    return np.array(blobs)

def MatchingEvaluation(detect_function, SURF):
    detect_fun = lambda I: detect_function(I)
    
    if SURF:
        desc_fun = lambda I, kp: p3.featuresSURF(I, kp)
    else:
        desc_fun = lambda I, kp: p3.featuresHOG(I, kp) 
        
    avg_scale_errors, avg_theta_errors = p3.matching_evaluation(detect_fun, desc_fun)

    for i in range(3):
        print('Avg. Scale Error for Image {}: {:.3f}'.format(i, avg_scale_errors[i]))
        print('Avg. Theta Error for Image {}: {:.3f}'.format(i, avg_theta_errors[i]))
    
def Scale_Theta_Errors_for_every_combination():
    detect_functions = [CornerDetect, MultiscaleCornerDetect, BlobDetect, MultiscaleBlobDetect, MultiscaleBlobDetectWithBoxFilters]
    flags = [True, False]
    for flag in flags:
        if flag:
            print("For SURF as local desciptor:")
            print()
        else:
            print("For HOG as local desciptor:")
            print()
        for detect_function in detect_functions:
            print('With ' + str(detect_function.__name__) + ' as a detect function:')
            MatchingEvaluation(detect_function, flag)
            print()
            
def Model_Training_and_Evaluation(detect_function, SURF):
    detect_fun = lambda I: detect_function(I)
    if SURF:
        a = 'SURF'
        desc_fun = lambda I, kp: p3.featuresSURF(I, kp)
    else:
        a = 'HOG'
        desc_fun = lambda I, kp: p3.featuresHOG(I, kp)
        
    feats = p3.FeatureExtraction(detect_fun, desc_fun)
    
    accs = []
    for k in range(5):
        x_train, y_train, x_test, y_test = p3.createTrainTest(feats, k)
    
        BOF_train, BOF_test = p3.BagOfWords(x_train, x_test)
        acc, preds, probas = p3.svm(BOF_train, y_train, BOF_test, y_test)
        accs.append(acc)
        
    print('Mean accuracy for ' + str(detect_function.__name__) + ' with ' + a + ' descriptors: {:.3f}%'.format(100.0*np.mean(accs)))
    
def Metric_Extraction_for_every_multiscale_detector():
    detect_functions = [MultiscaleCornerDetect, MultiscaleBlobDetect, MultiscaleBlobDetectWithBoxFilters]
    flags = [True, False]
    for flag in flags:
        if flag:
            print("For SURF as local desciptor:")
            print()
        else:
            print("For HOG as local desciptor:")
            print()
        for detect_function in detect_functions:
            print('With ' + str(detect_function.__name__) + ' as a detect function:')
            Model_Training_and_Evaluation(detect_function, flag)
            print()
            
img = cv2.imread("data_part12/edgetest_22.png", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float64)/255
# =============================================================================
# plt.imshow(img, cmap='gray')
# print(img.shape)
# 
# =============================================================================
# =============================================================================
# Imax = img.max()
# Imin = img.min()
# PSNR1 = 20
# PSNR2 = 10
# std1 = (Imax - Imin)/10**(PSNR1/20)
# std2 = (Imax - Imin)/10**(PSNR2/20)
# n1 = np.random.normal(0, std1, size = (img.shape[0], img.shape[1]))
# n2 = np.random.normal(0, std2, size = (img.shape[0], img.shape[1]))
# =============================================================================
# =============================================================================
# plt.figure()
# plt.imshow(img+n1, cmap = 'gray')
# plt.figure()
# plt.imshow(img+n2, cmap = 'gray')
# =============================================================================

# =============================================================================
# n_img = img + n1
# D = EdgeDetect(n_img, 2, linear = False)
# =============================================================================
# =============================================================================
# plt.figure()
# plt.imshow(D, 'gray')
# =============================================================================

# =============================================================================
# B = np.array([
#     [0,1,0],
#     [1,1,1],
#     [0,1,0]
# ], dtype=np.uint8)
# 
# M = cv2.dilate(img, B) - cv2.erode(img, B)
# #plt.figure()
# #plt.imshow(M, 'gray')
# T = M > 0.1
# # =============================================================================
# # plt.figure()
# # plt.imshow(T, 'gray')
# # =============================================================================
# 
# counter = 0
# 
# for i in range(D.shape[0]):
#     for j in range(D.shape[1]):
#         if D[i][j] == 1 and T[i][j] == 1:
#             counter += 1
#             
# precision = counter/T.sum()
# recall = counter/D.sum()
# C = (precision + recall)/2
# print (precision, recall, C)
# =============================================================================

colored_duomo = cv2.imread("data_part12/duomo_edges.jpg")
colored_duomo = cv2.cvtColor(colored_duomo, cv2.COLOR_BGR2RGB)

duomo = cv2.cvtColor(colored_duomo, cv2.COLOR_RGB2GRAY)
duomo = duomo.astype(np.float64)/255

# =============================================================================
# plt.figure()
# plt.imshow(duomo, 'gray')
# =============================================================================

# =============================================================================
# D2 = EdgeDetect(duomo, 2, linear = True)
# plt.figure()
# plt.imshow(D2, 'gray')
# 
# M2 = cv2.dilate(duomo, B) - cv2.erode(duomo, B)
# #plt.figure()
# #plt.imshow(M, 'gray')
# T2 = M2 > 0.1
# plt.figure()
# plt.imshow(T2, 'gray')
# 
# counter = 0
# 
# for i in range(D2.shape[0]):
#     for j in range(D2.shape[1]):
#         if D2[i][j] == 1 and T2[i][j] == 1:
#             counter += 1
#             
# precision = counter/T2.sum()
# recall = counter/D2.sum()
# C2 = (precision + recall)/2
# print (precision, recall, C2)
# =============================================================================

colored_donuts = cv2.imread("data_part12/donuts.jpg")
colored_donuts = cv2.cvtColor(colored_donuts, cv2.COLOR_BGR2RGB)

donuts = cv2.cvtColor(colored_donuts, cv2.COLOR_RGB2GRAY)
donuts = donuts.astype(np.float64)/255

colored_cells = cv2.imread("data_part12/cells.jpg")
colored_cells = cv2.cvtColor(colored_cells, cv2.COLOR_BGR2RGB)

cells = cv2.cvtColor(colored_cells, cv2.COLOR_RGB2GRAY)
cells = cells.astype(np.float64)/255

# =============================================================================
# interest_points = CornerDetect(duomo, s = 2)
# interest_points_visualization(colored_duomo, interest_points)
# 
# multiscale_interest_points = MultiscaleCornerDetect(duomo)
# interest_points_visualization(colored_duomo, multiscale_interest_points)
# =============================================================================

# =============================================================================
# blobs = BlobDetect(cells, s = 2)
# interest_points_visualization(colored_cells, blobs)
# 
# multiscale_blobs = MultiscaleBlobDetect(cells)
# interest_points_visualization(colored_cells, multiscale_blobs)
# =============================================================================

# =============================================================================
# box_blobs = BlobDetectWithBoxFilters(donuts)
# interest_points_visualization(colored_donuts, box_blobs)
# 
# multiscale_box_blobs = MultiscaleBlobDetectWithBoxFilters(donuts)
# interest_points_visualization(colored_donuts, multiscale_box_blobs)
# =============================================================================

#Scale_Theta_Errors_for_every_combination()

#Metric_Extraction_for_every_multiscale_detector()




















