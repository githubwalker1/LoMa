import cv2
from loma import LoMa, LoMaB128

model = LoMa(LoMaB128())

kptsA, kptsB = model.match(
    "/path/to/image_A.jpg",
    "/path/to/image_B.jpg",
)

F, mask = cv2.findFundamentalMat(
    kptsA,
    kptsB,
    method=cv2.USAC_MAGSAC,
    ransacReprojThreshold=0.5,
    confidence=0.999999,
    maxIters=10000,
)

inliers = int(mask.sum()) if mask is not None else 0
print("matches:", len(kptsA))
print("inliers:", inliers)
print("inlier ratio:", inliers / max(len(kptsA), 1))