"""
visual_odometry.py — ORB-based camera motion estimation fused with IMU.

Forward/backward detection uses OPTICAL FLOW DIRECTION:
  - When moving FORWARD, features spread outward from image center
    (Focus of Expansion). The dot product of (feature - center) and
    (flow vector) is POSITIVE.
  - When moving BACKWARD, features converge toward image center
    (Focus of Contraction). The dot product is NEGATIVE.

This gives us the sign that recoverPose alone cannot provide.
"""

import cv2, numpy as np, math

class VisualOdometry:
    def __init__(self, focal_length=300, cx=160, cy=120):
        self.focal = focal_length
        self.cx = cx
        self.cy = cy
        self.K = np.array([[focal_length,0,cx],[0,focal_length,cy],[0,0,1]], dtype=np.float64)
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.prev_kp = None
        self.prev_des = None
        self.x = 0.0; self.y = 0.0; self.theta = 0.0
        self.scale = 0.005
        self.matches_count = 0; self.inliers_count = 0
        self.direction = 1.0  # +1 = forward, -1 = backward

    def _detect_direction(self, pts_prev, pts_curr):
        """
        Determine if the camera is moving forward or backward
        using radial optical flow analysis.

        For each matched feature pair:
          1. Compute vector from image center to the previous position
          2. Compute the flow vector (how the feature moved)
          3. Dot product: positive = feature moved AWAY from center (forward)
                          negative = feature moved TOWARD center (backward)

        Average the dot products over all matches. The sign gives direction.
        """
        center = np.array([self.cx, self.cy])

        # Vector from center to each previous feature position
        radial = pts_prev - center  # shape (N, 2)

        # Optical flow for each feature
        flow = pts_curr - pts_prev  # shape (N, 2)

        # Dot product: radial · flow for each feature
        # Positive = expanding (forward), negative = contracting (backward)
        dots = np.sum(radial * flow, axis=1)

        avg_dot = np.mean(dots)

        # Only flip direction if signal is strong enough (avoid noise)
        if abs(avg_dot) > 0.5:
            return 1.0 if avg_dot > 0 else -1.0
        else:
            # Very small motion — keep previous direction
            return self.direction

    def process_frame(self, frame, imu_gz=0.0, dt=0.033):
        if frame is None: return 0.0, 0.0, 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if des is None or len(des) < 5 or self.prev_des is None or len(self.prev_des) < 5:
            self.prev_kp, self.prev_des = kp, des
            self.matches_count = self.inliers_count = 0
            return 0.0, 0.0, 0.0

        try:
            raw = self.matcher.knnMatch(self.prev_des, des, k=2)
        except cv2.error:
            self.prev_kp, self.prev_des = kp, des
            return 0.0, 0.0, 0.0

        good = [m for m,n in [p for p in raw if len(p)==2] if m.distance < 0.75*n.distance]
        self.matches_count = len(good)
        dx, dy, dtheta = 0.0, 0.0, 0.0

        if len(good) >= 10:
            p1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
            p2 = np.float32([kp[m.trainIdx].pt for m in good])

            # Detect forward vs backward from optical flow
            self.direction = self._detect_direction(p1, p2)

            E, mask = cv2.findEssentialMat(p1, p2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            if E is not None and mask is not None and E.shape == (3,3):
                self.inliers_count = int(mask.sum())
                if self.inliers_count >= 8:
                    _, R, t, _ = cv2.recoverPose(E, p1, p2, self.K, mask=mask)
                    tx, tz = float(t[0,0]), float(t[2,0])
                    vis_yaw = math.atan2(float(R[1,0]), float(R[0,0]))
                    imu_dt = imu_gz * (math.pi/180.0) * dt
                    dtheta = 0.3 * vis_yaw + 0.7 * imu_dt

                    # Apply direction sign to forward motion
                    # abs(tz) gives magnitude, self.direction gives sign
                    fwd = abs(tz) * self.scale * self.direction
                    lat = tx * self.scale

                    dx = fwd*math.cos(self.theta) - lat*math.sin(self.theta)
                    dy = fwd*math.sin(self.theta) + lat*math.cos(self.theta)
                else:
                    dtheta = imu_gz*(math.pi/180.0)*dt; self.inliers_count = 0
            else:
                dtheta = imu_gz*(math.pi/180.0)*dt; self.inliers_count = 0
        else:
            dtheta = imu_gz*(math.pi/180.0)*dt; self.inliers_count = 0

        self.theta += dtheta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        self.x += dx; self.y += dy
        self.prev_kp, self.prev_des = kp, des
        return dx, dy, dtheta

    def get_heading_deg(self):
        return (self.theta * 180.0 / math.pi) % 360
