import cv2
import numpy as np

def gray_world_white_balance(img):
    img = img.astype(np.float32)
    avg_b = np.mean(img[:,:,0])
    avg_g = np.mean(img[:,:,1])
    avg_r = np.mean(img[:,:,2])
    avg = (avg_b + avg_g + avg_r) / 3.0
    img[:,:,0] = np.clip(img[:,:,0] * (avg / (avg_b+1e-8)), 0, 255)
    img[:,:,1] = np.clip(img[:,:,1] * (avg / (avg_g+1e-8)), 0, 255)
    img[:,:,2] = np.clip(img[:,:,2] * (avg / (avg_r+1e-8)), 0, 255)
    return img.astype(np.uint8)

def color_correction_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2,a,b))
    corrected = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return corrected

def preprocess_image(path, target_max_dim=1600):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not open {path}")
    h, w = img.shape[:2]
    scale = 1.0
    max_dim = max(h, w)
    if max_dim > target_max_dim:
        scale = target_max_dim / max_dim
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    wb = gray_world_white_balance(img)
    den = cv2.fastNlMeansDenoisingColored(wb, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    bil = cv2.bilateralFilter(den, d=9, sigmaColor=75, sigmaSpace=75)
    corr = color_correction_clahe(bil)
    gauss = cv2.GaussianBlur(corr, (0,0), sigmaX=3)
    sharp = cv2.addWeighted(corr, 1.5, gauss, -0.5, 0)
    return sharp

def create_feature_detector():
    try:
        sift = cv2.SIFT_create()
        return 'SIFT', sift
    except Exception:
        orb = cv2.ORB_create(nfeatures=5000)
        return 'ORB', orb

def match_descriptors(desc1, desc2, method_name):
    if method_name == 'SIFT':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
    return matches

def ratio_test_filter(matches, ratio=0.75):
    good = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def compute_homography_inliers(kp1, kp2, good_matches, reproj_thresh=4.0):
    if len(good_matches) < 4:
        return None, []
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    if mask is None:
        return H, []
    inliers = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 1]
    return H, inliers

def visualize_matches(imgA, imgB, kp1, kp2, good_matches, inliers, probability):
    # we draw all the good matches
    img_all = cv2.drawMatches(imgA, kp1, imgB, kp2, good_matches, None,
                              matchColor=(180,180,180), singlePointColor=(255,0,0), flags=2)
    # we only draw inliers
    img_inliers = cv2.drawMatches(imgA, kp1, imgB, kp2, inliers, None,
                                  matchColor=(0,255,0), singlePointColor=None, flags=2)

    # text with probability
    cv2.putText(img_inliers, f"P = {probability:.3f}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)

    # showing results
    cv2.imshow("All matches (gray)", img_all)
    cv2.imshow("Inliers (green) - Final Identification", img_inliers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compute_identification_probability(pathA, pathB, params=None, verbose=True):
    if params is None:
        params = {}
    imgA = preprocess_image(pathA, target_max_dim=params.get('target_max_dim', 1600))
    imgB = preprocess_image(pathB, target_max_dim=params.get('target_max_dim', 1600))

    method_name, detector = create_feature_detector()
    if verbose:
        print("Feature detector:", method_name)

    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detector.detectAndCompute(grayA, None)
    kp2, des2 = detector.detectAndCompute(grayB, None)

    if verbose:
        print(f"Keypoints A: {len(kp1)}, Keypoints B: {len(kp2)}")

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        if verbose:
            print("Not enough descriptors for comparison.")
        return None

    raw_matches = match_descriptors(des1, des2, method_name)
    good_matches = ratio_test_filter(raw_matches, ratio=params.get('ratio', 0.75))

    H, inliers = compute_homography_inliers(kp1, kp2, good_matches, reproj_thresh=params.get('reproj_thresh', 4.0))
    usable = len(good_matches)
    num_inliers = len(inliers)
    prob = float(num_inliers) / usable if usable > 0 else 0.0

    if verbose:
        print(f"Good matches: {usable}")
        print(f"Inliers: {num_inliers}")
        print(f"Identification probability: {prob:.4f}")

    visualize_matches(imgA, imgB, kp1, kp2, good_matches, inliers, prob)

    return {
        'probability': prob,
        'num_good_matches': usable,
        'num_inliers': num_inliers,
    }

if __name__ == "__main__":
    pathA = "landset.png"
    pathB = "bing.png"

    result = compute_identification_probability(pathA, pathB, params={
        'target_max_dim': 1600,
        'ratio': 0.64,
        'reproj_thresh': 4.0
    }, verbose=True)

    if result:
        print("\nResult:")
        for k, v in result.items():
            if k == 'H' and v is not None:
                print(f"{k}: homography matrix, shape={v.shape}")
            else:
                print(f"{k}: {v}")

