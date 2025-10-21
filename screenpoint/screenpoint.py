import numpy as np
import cv2
import logging

MIN_MATCH_COUNT = 6
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6


def project(view, screen, debug=False, algorithm='orb'):
    """
    Project the centroid of a view image onto screen coordinates.

    Args:
        view: Grayscale view image (camera photo of screen)
        screen: Grayscale screen image (reference image)
        debug: If True, returns debug visualization image
        algorithm: Feature detection algorithm ('orb' or 'sift')
                  'orb' is 10-100x faster, 'sift' is more robust

    Returns:
        (x, y) coordinates, or (x, y, debug_img) if debug=True
        Returns (-1, -1) if matching fails
    """
    # Choose feature detector
    if algorithm.lower() == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
    else:  # default to ORB
        detector = cv2.ORB_create(nfeatures=2000)

    # Detect and compute features
    kp_screen, des_screen = detector.detectAndCompute(screen, None)
    kp_view, des_view = detector.detectAndCompute(view, None)

    # Choose matcher based on descriptor type
    if algorithm.lower() == 'sift':
        # SIFT uses floating-point descriptors, use FLANN with KDTree
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # ORB uses binary descriptors, use FLANN with LSH (Locality Sensitive Hashing)
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
                           key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches = matcher.knnMatch(des_screen, des_view, k=2)

    # Store all good matches as per Lowe's ration test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < MIN_MATCH_COUNT:
        logging.debug("ScreenPoint: Not enough matches.")
        return -1, -1

    screen_pts = np.float32([kp_screen[m.queryIdx].pt
                             for m in good]).reshape(-1, 1, 2)
    view_pts = np.float32([kp_view[m.trainIdx].pt
                           for m in good]).reshape(-1, 1, 2)

    h, w = view.shape
    M, mask = cv2.findHomography(view_pts, screen_pts, cv2.RANSAC, 5.0)

    pts = np.float32([[(w - 1) * 0.5, (h - 1) * 0.5]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    x, y = np.int32(dst[0][0])

    if debug:
        img_debug = draw_debug_(x, y, view, screen, M, mask, kp_screen,
                                kp_view, good)
        return x, y, img_debug
    else:
        return x, y


def draw_debug_(x, y, view, screen, M, mask, kp_screen, kp_view, good):

    matchesMask = mask.ravel().tolist()
    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2)
    img_debug = cv2.drawMatches(screen, kp_screen, view, kp_view, good, None,
                                **draw_params)

    # Get view centroid coordinates in img_debug space.
    cx = int(view.shape[1] * 0.5 + screen.shape[1])
    cy = int(view.shape[0] * 0.5)
    # Draw view outline.
    cv2.rectangle(img_debug, (screen.shape[1], 0),
                  (img_debug.shape[1] - 2, img_debug.shape[0] - 2),
                  (0, 0, 255), 2)
    # draw connecting line.
    cv2.polylines(img_debug, [np.int32([(x, y), (cx, cy)])], True,
                  (100, 100, 255), 1, cv2.LINE_AA)
    # Draw query/match markers.
    cv2.drawMarker(img_debug, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.circle(img_debug, (x, y), 10, (0, 0, 255), -1)

    return img_debug