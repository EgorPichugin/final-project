#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===================================================================================
  Lane Detection in CARLA Simulator - ADC Course
===================================================================================

  This script connects to the CARLA simulator, spawns an ego vehicle with camera
  sensors, and performs real-time lane detection on the live RGB camera feed.

  ---- PIPELINE OVERVIEW ----
  This version intentionally mimics the simple classroom OpenCV example:
  1. Read the current RGB camera frame from CARLA.
  2. Convert BGR/RGB image data into grayscale.
  3. Apply Gaussian blur to reduce noise.
  4. Apply Canny edge detection.
  5. Keep only the lane area using a triangular Region of Interest (ROI).
  6. Detect line segments using Probabilistic Hough Transform.
  7. Draw the detected line segments on a blank image.
  8. Blend the detected lines with the original camera image.

  A commented duplicate student-exercise block is included inside the HUD class.
  You can remove the implemented function during the lecture and ask students to
  complete the placeholder version step by step.

  ---- KEYBOARD CONTROLS ----
  W / S        = Throttle / Brake
  A / D        = Steer left / right
  Z            = Toggle reverse gear
  H            = Hand brake
  F1           = Restart (respawn vehicle)
  F5           = Toggle autopilot on/off
  TAB          = Toggle camera view (front / rear)
  1-9 / 0      = Switch sensor type (RGB, Depth, Segmentation, etc.)
  F9           = Toggle image recording
  F11 / F12    = Cycle weather presets backward / forward
  ESC          = Quit

  ---- REQUIREMENTS ----
  - CARLA Server (CarlaUE4.exe) running on localhost:2000
  - Python 3.7+
  - numpy, opencv-python (cv2)
===================================================================================
"""

# ================================================================================
# IMPORTS
# ================================================================================
import glob       # Used to find the CARLA .egg file on disk
import os         # Operating system utilities (path joining, OS detection)
import sys        # System-level utilities (path manipulation, version info)
import re         # Regular expressions for parsing weather preset names
import random     # Random selection of blueprints, colors, spawn points
import time       # Used for timestamping keyboard input
import weakref    # Weak references to avoid circular references in callbacks
import math       # Mathematical functions (not heavily used but available)

import numpy as np  # Numerical array operations (image processing, polynomial fitting)
import cv2          # OpenCV for image display, morphological ops, drawing

# ================================================================================
# LOCATE AND IMPORT THE CARLA PYTHON API
# ================================================================================
# CARLA distributes its Python client as a .egg file. We need to add it to
# sys.path so that "import carla" works. The .egg filename encodes the Python
# version and OS, so we use glob to find the correct one automatically.
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,          # e.g. 3
        sys.version_info.minor,          # e.g. 7
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass  # .egg not found via glob; carla may already be installed via pip

# Fallback: also add the PythonAPI/carla directory relative to this script's location.
# This covers the case when running from the PythonAPI/examples/ folder directly.
try:
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla'
    )
except IndexError:
    pass

import carla                          # The CARLA Python client library
from carla import ColorConverter as cc  # Shortcuts for sensor color conversion modes


# ================================================================================
# CONFIGURATION (mc_args)
# ================================================================================
class Config:
    """
    Central configuration for the script. Instead of using argparse (which
    requires command-line flags), we hardcode sensible defaults here so
    students can simply run the script.

    Attributes
    ----------
    host : str
        IP address of the machine running the CARLA server.
        '127.0.0.1' means localhost (same machine).
    port : int
        TCP port the CARLA server listens on (default 2000).
    autopilot : bool
        If True, the vehicle starts in autopilot mode (Traffic Manager drives).
        If False, you must drive manually with WASD keys.
    width : int
        Horizontal resolution (pixels) for all camera sensors.
    height : int
        Vertical resolution (pixels) for all camera sensors.
    """
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 2000
        self.autopilot = True
        self.width = 460
        self.height = 360


# ================================================================================
# KEYBOARD CONTROL
# ================================================================================
class KeyboardControl:
    """
    Handles keyboard input from the OpenCV window and translates key presses
    into CARLA vehicle control commands or simulation actions.

    The OpenCV function cv2.waitKeyEx() returns an integer keycode each frame.
    This class maps those keycodes to actions like steering, throttle, or
    toggling autopilot.

    Attributes
    ----------
    _autopilot_enabled : bool
        Whether the vehicle is currently in autopilot mode.
    _control : carla.VehicleControl
        The current control state (throttle, steer, brake, reverse, hand_brake).
    _steer_cache : float
        Accumulated steering value for smooth transitions when holding A/D.
    """

    def __init__(self, world, start_in_autopilot):
        """
        Parameters
        ----------
        world : World
            The World object that holds the ego vehicle and sensors.
        start_in_autopilot : bool
            If True, the vehicle begins driving itself via the Traffic Manager.
        """
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        # Tell the CARLA server to enable/disable autopilot on this vehicle
        world.vehicle.set_autopilot(self._autopilot_enabled)

    def parse_events(self, world, key):
        """
        Respond to a single keypress from the current frame.

        This method is called once per frame in the main loop. It checks
        the keycode and performs the appropriate action:

        - F1  (65470)  : Restart the simulation (respawn vehicle + sensors)
        - TAB (9)      : Toggle between front and rear camera positions
        - F5  (65474)  : Toggle autopilot on/off
        - F9  (65478)  : Toggle image recording to disk
        - F11 (65480)  : Previous weather preset
        - F12 (65481)  : Next weather preset
        - 0-9          : Switch to a specific sensor type
        - Z            : Toggle reverse gear

        If autopilot is OFF, it also calls _parse_drive_keys() to handle
        W/A/S/D/H for manual driving, and applies the resulting control
        to the vehicle.

        Parameters
        ----------
        world : World
            The simulation world containing the vehicle.
        key : int
            The keycode returned by cv2.waitKeyEx().
        """
        # --- Simulation control keys ---
        if key == 65470:                        # F1: Restart
            world.restart()
        elif key == 9:                           # TAB: Toggle camera view
            world.camera_manager.toggle_camera()
        elif key == 65480:                       # F11: Previous weather
            world.next_weather(reverse=True)
        elif key == 65481:                       # F12: Next weather
            world.next_weather()
        elif key == ord('0'):                    # 0: Cycle to next sensor
            world.camera_manager.next_sensor()
        elif ord('1') <= key <= ord('9'):         # 1-9: Jump to specific sensor
            world.camera_manager.set_sensor(key - ord('1'))
        elif key == 65478:                       # F9: Toggle recording
            world.camera_manager.toggle_recording()
        elif key == ord('z'):                    # Z: Toggle reverse
            self._control.reverse = not self._control.reverse
        elif key == 65474:                       # F5: Toggle autopilot
            self._autopilot_enabled = not self._autopilot_enabled
            world.vehicle.set_autopilot(self._autopilot_enabled)
            world.hud.notification(
                'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

        # --- Manual driving (only when autopilot is OFF) ---
        if not self._autopilot_enabled:
            self._parse_drive_keys(key, int(round(time.time() * 1000)))
            world.vehicle.apply_control(self._control)

    def _parse_drive_keys(self, key, milliseconds):
        """
        Translate W/A/S/D/H keys into vehicle control values.

        Steering is accumulated over time (while A or D is held) for a
        smooth feel, clamped to [-0.7, 0.7] to prevent extreme turns.

        Parameters
        ----------
        key : int
            The keycode of the currently pressed key.
        milliseconds : int
            Current time in ms, used to scale the steering increment so
            that steering speed is frame-rate independent.
        """
        # Throttle: full (1.0) when W is pressed, otherwise 0
        self._control.throttle = 1.0 if key == ord('w') else 0.0

        # Steering: accumulate left/right while A/D are held
        steer_increment = 5e-4 * milliseconds
        if key == ord('a'):
            self._steer_cache -= steer_increment   # Steer left (negative)
        elif key == ord('d'):
            self._steer_cache += steer_increment   # Steer right (positive)
        else:
            self._steer_cache = 0.0                 # Release: center steering

        # Clamp steering to [-0.7, 0.7] range
        self._steer_cache = max(-0.7, min(0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)

        # Brake: full (1.0) when S is pressed
        self._control.brake = 1.0 if key == ord('s') else 0.0

        # Hand brake: engaged when H is pressed
        self._control.hand_brake = (key == ord('h'))


# ================================================================================
# HUD (Head-Up Display) + LANE DETECTION
# ================================================================================
class HUD:
    """
    The HUD class serves two purposes:
    1. Manages OpenCV display windows for the RGB and processed images.
    2. Contains the entire lane detection pipeline:
       - Semantic mask extraction (Green == 234)
       - ROI (Region of Interest) masking
       - Morphological dilation
       - Sliding-window lane pixel search
       - Polynomial curve fitting with temporal smoothing
       - Lane overlay drawing
       - Curvature and offset computation

    Attributes
    ----------
    dim : tuple of (int, int)
        Image dimensions (width, height).
    name : str
        Label suffix for OpenCV window titles.
    left_a, left_b, left_c : list of float
        History of polynomial coefficients (a*y^2 + b*y + c) for the left lane.
        Used for temporal smoothing over the last N frames.
    right_a, right_b, right_c : list of float
        Same as above but for the right lane.
    """

    # Maximum number of past frames to average polynomial coefficients over.
    # Higher = smoother but slower to react to real lane changes.
    SMOOTHING_WINDOW = 100

    def __init__(self, width, height, name="raw"):
        """
        Parameters
        ----------
        width : int
            Image width in pixels.
        height : int
            Image height in pixels.
        name : str
            Label for OpenCV window titles (e.g. "raw").
        """
        self.dim = (width, height)
        self.name = name
        self._notification_text = 'Autopilot off'

        # History buffers for polynomial smoothing.
        # Each frame we append the latest a, b, c coefficients and average
        # over the last SMOOTHING_WINDOW entries. This prevents jittery lanes.
        self.left_a, self.left_b, self.left_c = [], [], []
        self.right_a, self.right_b, self.right_c = [], [], []
        self.frame_count = 0
        self.images_saved = False

    # ---- Notifications (simple text overlay system) ----

    def notification(self, text, seconds=2.0):
        """Store a notification string (displayed in terminal for now)."""
        self._notification_text = text

    def error(self, text):
        """Store an error string (same mechanism as notification)."""
        self._notification_text = text

    def tick(self, world, clock):
        """Called each frame. Reserved for future HUD updates (currently unused)."""
        pass

    # ---- Lane Detection Pipeline ----

    def _extract_lane_mask(self, semantic_image):
        """
        Extract a binary mask of lane-line pixels from the semantic camera image.

        In CARLA's CityScapes palette, road lane markings have the color
        (B=50, G=234, R=157). We detect them by checking Green == 234.

        Parameters
        ----------
        semantic_image : np.ndarray, shape (H, W, 3), dtype uint8
            The semantic segmentation image in BGR format.

        Returns
        -------
        mask : np.ndarray, shape (H, W), dtype uint8
            Binary mask: 255 where lane lines are detected, 0 elsewhere.
        """
        # Check only the Green channel (index 1 in BGR) for the value 234
        mask = (semantic_image[:, :, 1] == 234).astype(np.uint8) * 255
        return mask

    def _apply_roi(self, mask, image_shape):
        """
        Apply a Region of Interest (ROI) mask to keep only the road area
        in front of the vehicle.

        We define a hexagonal polygon that covers the lower portion of the
        image (where the road is) and blacks out everything else (sky,
        buildings, trees on the sides).

        Parameters
        ----------
        mask : np.ndarray, shape (H, W), dtype uint8
            The binary lane mask.
        image_shape : tuple of (H, W, C)
            Shape of the original image, used to compute polygon vertices.

        Returns
        -------
        masked : np.ndarray, shape (H, W), dtype uint8
            The lane mask with only the ROI region preserved.
        roi_visual : np.ndarray, shape (H, W, 3), dtype uint8
            A 3-channel visualization of the ROI polygon (for debugging display).
        """
        height, width = image_shape[:2]

        # Define 5 vertices of a trapezoidal/hexagonal polygon.
        # Bottom-left -> mid-left -> top-center -> mid-right -> bottom-right
        hexagon = np.array([[
            (0, height),                              # bottom-left corner
            (0, int(height / 2) + 100),               # left edge, midway
            (int(width / 2), int(height / 2) - 10),   # top center (vanishing pt)
            (width, int(height / 2) + 100),           # right edge, midway
            (width, height)                            # bottom-right corner
        ]])

        # Create a 3-channel ROI mask for visualization
        roi_visual = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.fillPoly(roi_visual, hexagon, (255, 255, 255))

        # Create a single-channel ROI mask and apply it via bitwise AND
        roi_single = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(roi_single, hexagon, 255)
        masked = cv2.bitwise_and(mask, roi_single)

        return masked, roi_visual

    def _dilate_mask(self, mask):
        """
        Apply morphological dilation to thicken the lane line pixels.

        Lane markings in the semantic image are often only 1-2 pixels wide,
        which makes sliding-window detection unreliable. Dilation expands
        each white pixel outward using a rectangular kernel.

        Parameters
        ----------
        mask : np.ndarray, shape (H, W), dtype uint8
            Binary lane mask.

        Returns
        -------
        dilated : np.ndarray, shape (H, W), dtype uint8
            Dilated lane mask with thicker lane markings.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return dilated

    def lanes_detection(self, binary_img, original_img,
                        nwindows=40, margin=10, minpix=1):
        """
        Core lane detection using the sliding-window method.

        ALGORITHM:
        1. Take a histogram of the bottom half of the binary image.
           The two peaks (left half and right half) indicate the base
           x-positions of the left and right lane lines.
        2. Divide the image into `nwindows` horizontal strips (windows).
        3. Starting from the bottom, search for nonzero (white) pixels
           within a window of width 2*margin centered on the current
           lane x-position.
        4. If enough pixels are found (>= minpix), recenter the window
           on their mean x-position for the next strip up.
        5. After scanning all windows, collect all the lane pixel
           coordinates and fit a 2nd-degree polynomial: x = a*y^2 + b*y + c
        6. Average the polynomial coefficients over the last N frames
           for temporal smoothing.
        7. Evaluate the polynomials to generate smooth lane curves, then
           draw the lane area overlay on the original image.

        Parameters
        ----------
        binary_img : np.ndarray, shape (H, W), dtype uint8
            The dilated binary lane mask.
        original_img : np.ndarray, shape (H, W, 3), dtype uint8
            The original RGB camera image (for overlay drawing).
        nwindows : int
            Number of sliding windows stacked vertically.
            More windows = finer vertical resolution but slower.
        margin : int
            Half-width of each sliding window in pixels.
            Larger = captures more pixels but may merge adjacent lanes.
        minpix : int
            Minimum number of pixels in a window required to recenter
            the window position. Prevents noise from shifting the window.

        Returns
        -------
        lane_overlay : np.ndarray, shape (H, W, 3), dtype uint8
            The original image with the detected lane area drawn on top.
        sliding_window_img : np.ndarray, shape (H, W, 3), dtype uint8
            Debug visualization showing the sliding windows and detected pixels.
        """
        # Convert single-channel binary to 3-channel for colored visualization
        if len(binary_img.shape) == 2:
            vis_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = binary_img.copy()

        height, width = binary_img.shape[:2]

        # --- Step 1: Histogram to find lane base positions ---
        # Sum all white pixels in each column of the bottom half.
        # The column with the most white pixels on each side is the lane base.
        if len(binary_img.shape) == 2:
            histogram = np.sum(binary_img[height // 2:, :], axis=0)
        else:
            gray = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
            histogram = np.sum(gray[height // 2:, :], axis=0)

        midpoint = width // 2
        leftx_base = np.argmax(histogram[:midpoint])      # Left lane base x
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint  # Right lane base x

        # --- Step 2: Sliding window setup ---
        window_height = height // nwindows     # Height of each window strip

        # Find all nonzero (white) pixel coordinates in the binary image
        if len(binary_img.shape) == 2:
            nonzero_y, nonzero_x = binary_img.nonzero()
        else:
            gray = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
            nonzero_y, nonzero_x = gray.nonzero()

        # Current x position for each lane, updated as we scan upward
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Lists to accumulate pixel indices belonging to each lane
        left_lane_indices = []
        right_lane_indices = []

        # --- Step 3-4: Scan windows from bottom to top ---
        out_img = vis_img.copy()
        for win in range(nwindows):
            # Vertical boundaries of this window
            win_y_bottom = height - (win + 1) * window_height
            win_y_top = height - win * window_height

            # Horizontal boundaries for left lane window
            win_xleft_lo = leftx_current - margin
            win_xleft_hi = leftx_current + margin
            # Horizontal boundaries for right lane window
            win_xright_lo = rightx_current - margin
            win_xright_hi = rightx_current + margin

            # Draw window rectangles for debug visualization
            cv2.rectangle(out_img,
                          (win_xleft_lo, win_y_bottom),
                          (win_xleft_hi, win_y_top),
                          (100, 255, 255), 1)
            cv2.rectangle(out_img,
                          (win_xright_lo, win_y_bottom),
                          (win_xright_hi, win_y_top),
                          (100, 255, 255), 1)

            # Find indices of nonzero pixels that fall inside the left window
            good_left = (
                (nonzero_y >= win_y_bottom) & (nonzero_y < win_y_top) &
                (nonzero_x >= win_xleft_lo) & (nonzero_x < win_xleft_hi)
            ).nonzero()[0]

            # Find indices of nonzero pixels that fall inside the right window
            good_right = (
                (nonzero_y >= win_y_bottom) & (nonzero_y < win_y_top) &
                (nonzero_x >= win_xright_lo) & (nonzero_x < win_xright_hi)
            ).nonzero()[0]

            left_lane_indices.append(good_left)
            right_lane_indices.append(good_right)

            # Recenter the window if enough pixels were found
            if len(good_left) >= minpix:
                leftx_current = int(np.mean(nonzero_x[good_left]))
            if len(good_right) >= minpix:
                rightx_current = int(np.mean(nonzero_x[good_right]))

        # --- Step 5: Concatenate and extract pixel coordinates ---
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        left_x = nonzero_x[left_lane_indices]    # x coords of left lane pixels
        left_y = nonzero_y[left_lane_indices]     # y coords of left lane pixels
        right_x = nonzero_x[right_lane_indices]   # x coords of right lane pixels
        right_y = nonzero_y[right_lane_indices]    # y coords of right lane pixels

        # Color the detected pixels on the debug image
        out_img[left_y, left_x] = [255, 0, 100]    # Left lane = magenta
        out_img[right_y, right_x] = [0, 100, 255]   # Right lane = orange

        # --- Step 6: Polynomial fitting with temporal smoothing ---
        try:
            # Fit: x = a*y^2 + b*y + c  (2nd degree polynomial)
            left_fit = np.polyfit(left_y, left_x, 2)
            right_fit = np.polyfit(right_y, right_x, 2)

            # Store coefficients in history buffers
            self.left_a.append(left_fit[0])
            self.left_b.append(left_fit[1])
            self.left_c.append(left_fit[2])
            self.right_a.append(right_fit[0])
            self.right_b.append(right_fit[1])
            self.right_c.append(right_fit[2])

            # Average over the last SMOOTHING_WINDOW frames
            N = self.SMOOTHING_WINDOW
            left_fit_avg = np.array([
                np.mean(self.left_a[-N:]),
                np.mean(self.left_b[-N:]),
                np.mean(self.left_c[-N:])
            ])
            right_fit_avg = np.array([
                np.mean(self.right_a[-N:]),
                np.mean(self.right_b[-N:]),
                np.mean(self.right_c[-N:])
            ])

            # --- Step 7: Evaluate polynomials and draw overlay ---
            plot_y = np.linspace(0, height - 1, height)
            left_fitx = (left_fit_avg[0] * plot_y ** 2 +
                         left_fit_avg[1] * plot_y +
                         left_fit_avg[2])
            right_fitx = (right_fit_avg[0] * plot_y ** 2 +
                          right_fit_avg[1] * plot_y +
                          right_fit_avg[2])

            # Compute curvature (informational, not displayed in this version)
            _ = self._get_curvature(height, width, plot_y, left_fitx, right_fitx)

            # Draw the lane area on the original image
            lane_img = self._draw_lane_area(height, width, plot_y,
                                            left_fitx, right_fitx)

            # Blend only the bottom 2/3 of the lane overlay onto the RGB image
            # (the top portion is beyond the vanishing point and looks bad)
            blend_start = height // 2 + height // 6
            overlay = np.zeros_like(original_img)
            overlay[blend_start:, :, :] = lane_img[blend_start:, :, :]
            lane_overlay = cv2.addWeighted(original_img, 1.0, overlay, 0.7, 0)

        except (TypeError, ValueError, np.linalg.LinAlgError):
            # If fitting fails (not enough pixels, singular matrix, etc.),
            # just return the original image without overlay
            lane_overlay = original_img.copy()

        return lane_overlay, out_img

    def _get_curvature(self, img_h, img_w, plot_y, left_fitx, right_fitx):
        """
        Compute the radius of curvature for each lane and the vehicle's
        lateral offset from lane center.

        The polynomial is in pixel space. We convert to real-world meters
        using approximate scale factors, then apply the curvature formula:

            R = (1 + (dy/dx)^2)^(3/2) / |d2y/dx2|

        For a polynomial x = a*y^2 + b*y + c:
            dx/dy = 2*a*y + b
            d2x/dy2 = 2*a

        Parameters
        ----------
        img_h, img_w : int
            Image height and width.
        plot_y : np.ndarray
            Array of y-coordinates spanning the full image height.
        left_fitx, right_fitx : np.ndarray
            The fitted x-coordinates for the left and right lanes.

        Returns
        -------
        tuple of (float, float, float)
            (left_curvature_radius, right_curvature_radius, center_offset)
            All in meters.
        """
        y_eval = np.max(plot_y)          # Evaluate curvature at the bottom of the image

        # Approximate conversion factors from pixels to meters.
        # These depend on camera FOV, resolution, and mounting height.
        ym_per_pix = 30.5 / 720          # ~30.5 meters of road visible vertically
        xm_per_pix = 3.7 / 720           # ~3.7 meters of lane width horizontally

        # Re-fit polynomials in real-world (meter) coordinates
        left_fit_m = np.polyfit(plot_y * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_m = np.polyfit(plot_y * ym_per_pix, right_fitx * xm_per_pix, 2)

        # Radius of curvature formula
        left_R = ((1 + (2 * left_fit_m[0] * y_eval * ym_per_pix +
                        left_fit_m[1]) ** 2) ** 1.5 /
                  np.absolute(2 * left_fit_m[0]))
        right_R = ((1 + (2 * right_fit_m[0] * y_eval * ym_per_pix +
                         right_fit_m[1]) ** 2) ** 1.5 /
                   np.absolute(2 * right_fit_m[0]))

        # Vehicle center offset from lane center
        car_center = img_w / 2
        left_bottom = (left_fit_m[0] * img_h ** 2 +
                       left_fit_m[1] * img_h + left_fit_m[2])
        right_bottom = (right_fit_m[0] * img_h ** 2 +
                        right_fit_m[1] * img_h + right_fit_m[2])
        lane_center = (left_bottom + right_bottom) / 2
        offset = (car_center - lane_center) * xm_per_pix / 10  # in meters

        return (left_R, right_R, offset)

    def _draw_lane_area(self, img_h, img_w, plot_y, left_fitx, right_fitx):
        """
        Create an image with the area between the two lane lines filled
        with a semi-transparent color.

        We build a polygon from the left lane points (top to bottom) and
        the right lane points (bottom to top, flipped), then fill it.

        Parameters
        ----------
        img_h, img_w : int
            Image dimensions.
        plot_y : np.ndarray
            Y-coordinates for the fitted curves.
        left_fitx, right_fitx : np.ndarray
            X-coordinates for the left and right lane curves.

        Returns
        -------
        color_img : np.ndarray, shape (H, W, 3), dtype uint8
            Image with the filled lane polygon.
        """
        color_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        # Build polygon: left lane going down, right lane going up
        left_points = np.array([np.transpose(np.vstack([left_fitx, plot_y]))])
        right_points = np.array([
            np.flipud(np.transpose(np.vstack([right_fitx, plot_y])))
        ])
        all_points = np.hstack((left_points, right_points))

        # Fill the polygon with a cyan/yellow color (BGR: 0, 200, 255)
        cv2.fillPoly(color_img, np.int_(all_points), (0, 200, 255))

        return color_img


    def _detect_lanes_simple_opencv(self, img_bgr):
        if img_bgr is None:
            return None, None, None, None, None, None, None

        original_img = img_bgr.copy()

        # Convert the RGB camera frame to grayscale.
        img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        # Reduce noise before edge detection.
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Detect strong intensity edges.
        edges = cv2.Canny(img_blur, 50, 150)

        height, width = edges.shape

        # Keep only the road region in front of the vehicle.
        roi_vertices = np.array([[
            (0, height),
            (width // 2, int(height * 0.55)),
            (width, height)
        ]], dtype=np.int32)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Detect straight lane-line segments.
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=25,
            maxLineGap=40
        )

        line_image = np.zeros_like(original_img)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
                cv2.line(
                    line_image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    3
                )

        # Overlay detected lines onto the original camera frame.
        final_result = cv2.addWeighted(
            original_img, 0.8,
            line_image, 1.0,
            0
        )

        return (
            final_result,
            img_gray,
            img_blur,
            edges,
            mask,
            masked_edges,
            line_image
        )

    def render(self, semantic_image, raw_image):
        """
        Main render function called each frame.

        This classroom version runs the same simple OpenCV lane detection pipeline
        on the live CARLA RGB camera feed. The semantic image is still accepted in
        the function signature because the World class already passes it, but the
        actual lane detection here does NOT depend on semantic segmentation.

        Display windows:
        - "1- Original RGB Camera"       : Live CARLA RGB camera frame
        - "2- Grayscale Image"           : Grayscale conversion
        - "3- Gaussian Blur"             : Blurred grayscale image
        - "4- Canny Edge Detection"      : Canny edges
        - "5- ROI Mask"                  : Triangular lane ROI
        - "6- Edges After ROI Masking"   : Canny edges restricted to ROI
        - "7- Detected Lane Lines"       : Hough line segments only
        - "8- Final Lane Detection"      : Final overlay on RGB frame

        Parameters
        ----------
        semantic_image : np.ndarray or None
            Kept for compatibility with the previous CARLA implementation.
            Not used by this simple OpenCV pipeline.
        raw_image : np.ndarray or None
            The RGB camera image as a BGR OpenCV array.

        Returns
        -------
        key : int
            The keycode from cv2.waitKeyEx() (used for keyboard controls).
        result : np.ndarray
            The final lane-overlay image.
        """
        result = raw_image

        if raw_image is not None:
            self.frame_count += 1

            if semantic_image is not None:
                # Extract lane pixels from semantic segmentation.
                segmented_lanes = self._extract_lane_mask(semantic_image)

                # Restrict detection to the road area.
                segmented_lanes, _ = self._apply_roi(
                    segmented_lanes,
                    raw_image.shape
                )

                # Improve thin lane pixels using dilation.
                improved_lanes = self._dilate_mask(segmented_lanes)

                # Edge detection from the raw RGB image.
                gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)

                # Save one stable frame after the simulation starts.
                if self.frame_count >= 100 and not self.images_saved:
                    os.makedirs("task1_results", exist_ok=True)

                    cv2.imwrite("task1_results/01_raw.png", raw_image)
                    cv2.imwrite("task1_results/02_semantic.png", semantic_image)
                    cv2.imwrite("task1_results/03_edges.png", edges)
                    cv2.imwrite("task1_results/04_segmented_lanes.png", segmented_lanes)
                    cv2.imwrite("task1_results/05_improved_lanes.png", improved_lanes)

                    self.images_saved = True
                    print("Task 1 images saved to task1_results/")
            (
                result,
                img_gray,
                img_blur,
                edges,
                roi_mask,
                masked_edges,
                line_image
            ) = self._detect_lanes_simple_opencv(raw_image)

            # Show each step exactly like the notebook/classroom explanation.
            cv2.imshow('1- Original RGB Camera', raw_image)
            cv2.imshow('2- Grayscale Image', img_gray)
            cv2.imshow('3- Gaussian Blur', img_blur)
            cv2.imshow('4- Canny Edge Detection', edges)
            cv2.imshow('5- ROI Mask', roi_mask)
            cv2.imshow('6- Edges After ROI Masking', masked_edges)
            cv2.imshow('7- Detected Lane Lines (Hough)', line_image)
            cv2.imshow('8- Final Lane Detection Result', result)

            # Optional: still show the semantic camera if it exists, but it is not
            # part of this simple lane detection implementation.
            if semantic_image is not None:
                cv2.imshow('Semantic Segmentation (optional reference)', semantic_image)

        # Wait for a keypress (30ms timeout = ~33 FPS max display rate)
        key = cv2.waitKeyEx(30)

        # ESC pressed: close all OpenCV windows
        if key == 27:
            cv2.destroyAllWindows()

        return key, result


# ================================================================================
# COLLISION SENSOR
# ================================================================================
class CollisionSensor:
    """
    Attaches a collision sensor to the ego vehicle and prints/notifies
    when a collision occurs.

    The collision sensor is invisible (no physical shape). It simply
    fires a callback whenever the parent actor physically contacts
    another actor in the simulation.

    Attributes
    ----------
    sensor : carla.Actor
        The collision sensor actor in the CARLA world.
    """

    def __init__(self, parent_actor, hud):
        """
        Parameters
        ----------
        parent_actor : carla.Actor
            The vehicle to attach the sensor to.
        hud : HUD or None
            If provided, collision messages appear as HUD notifications.
            Otherwise they are printed to the console.
        """
        self.sensor = None
        self._parent = parent_actor
        self._hud = hud

        # Find the collision sensor blueprint and spawn it attached to the vehicle
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

        # Register the callback using a weak reference to avoid preventing
        # garbage collection of this object (prevents memory leaks)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event)
        )

    @staticmethod
    def _on_collision(weak_self, event):
        """
        Callback fired when a collision occurs.

        Parameters
        ----------
        weak_self : weakref
            Weak reference to the CollisionSensor instance.
        event : carla.CollisionEvent
            Contains info about what was hit and the collision impulse.
        """
        self = weak_self()
        if not self:
            return
        # Format the other actor's type_id into a human-readable name
        # e.g. "static.prop.streetbarrier" -> "Prop Streetbarrier"
        actor_type = ' '.join(
            event.other_actor.type_id.replace('_', '.').title().split('.')[1:]
        )
        if self._hud is not None:
            self._hud.notification('Collision with %r' % actor_type)
        else:
            print('Collision with %r' % actor_type)


# ================================================================================
# CAMERA MANAGER
# ================================================================================
class CameraManager:
    """
    Manages a single camera sensor attached to the ego vehicle.

    Two instances are created in the World class:
    1. An RGB camera (sensor index 0) for the raw color feed.
    2. A Semantic Segmentation camera (sensor index 5) for lane detection.

    Each instance can switch between different sensor types (RGB, Depth,
    Segmentation with various color converters) and toggle between
    front and rear mounting positions.

    Attributes
    ----------
    sensor : carla.Actor
        The currently active camera sensor actor.
    _parent : carla.Actor
        The vehicle the camera is attached to.
    _image_raw : np.ndarray or None
        The latest image received from the sensor callback.
    _camera_transforms : list of carla.Transform
        Available mounting positions (front dashboard, rear overview).
    _sensors : list
        Definitions of available sensor types and their color converters.
    _index : int or None
        Index of the currently active sensor in _sensors.
    _recording : bool
        Whether images are being saved to disk.
    """

    def __init__(self, parent_actor, hud, config):
        """
        Parameters
        ----------
        parent_actor : carla.Actor
            The vehicle to attach cameras to.
        hud : HUD or None
            Used to read image dimensions and send notifications.
        config : Config
            Configuration object with width/height settings.
        """
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._image_raw = None
        self._index = None

        # Two camera mounting positions:
        # [0] Front dashboard: just above the hood, looking forward
        # [1] Rear overview:   behind and above the car, angled down 15 degrees
        self._camera_transforms = [
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(x=-5.5, z=2.8),
                            carla.Rotation(pitch=-15))
        ]
        self._transform_index = 1   # Start with rear view

        # Available sensor types with their color converter modes.
        # Each entry: [blueprint_id, ColorConverter, display_name]
        # A 4th element (the Blueprint object) is appended below after lookup.
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw,
             'Camera RGB'],
            ['sensor.camera.depth', cc.Raw,
             'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth,
             'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth,
             'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw,
             'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)']
        ]

        # Look up each sensor blueprint and configure its resolution
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if hud is not None:
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            else:
                bp.set_attribute('image_size_x', str(config.width))
                bp.set_attribute('image_size_y', str(config.height))
            item.append(bp)  # item is now [id, converter, name, Blueprint]

    def toggle_camera(self):
        """Switch between front and rear camera positions."""
        self._transform_index = (
            (self._transform_index + 1) % len(self._camera_transforms)
        )
        self.sensor.set_transform(
            self._camera_transforms[self._transform_index]
        )

    def set_sensor(self, index, notify=True):
        """
        Activate a specific sensor type by index.

        If the new sensor is a different blueprint than the current one,
        the old sensor is destroyed and a new one is spawned. If only
        the color converter differs (same blueprint), we just update
        the index without respawning.

        Parameters
        ----------
        index : int
            Index into self._sensors (0-5).
        notify : bool
            If True, display the sensor name as a HUD notification.
        """
        index = index % len(self._sensors)

        # Check if we need to spawn a new sensor (different blueprint type)
        needs_respawn = (
            self._index is None or
            self._sensors[index][0] != self._sensors[self._index][0]
        )

        if needs_respawn:
            # Destroy the old sensor if it exists
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None

            # Spawn the new sensor at the current camera position
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],                       # Blueprint object
                self._camera_transforms[self._transform_index],  # Transform
                attach_to=self._parent                           # Attach to vehicle
            )

            # Register the image callback with a weak reference
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image)
            )

        if notify and self._hud is not None:
            self._hud.notification(self._sensors[index][2])

        self._index = index

    def next_sensor(self):
        """Cycle to the next sensor type in the list."""
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        """Toggle saving images to disk on/off."""
        self._recording = not self._recording
        if self._hud is not None:
            self._hud.notification(
                'Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        """
        Return the latest image received from the sensor.

        Parameters
        ----------
        display : unused
            Kept for API compatibility (originally used for Pygame surface).

        Returns
        -------
        np.ndarray or None
            The latest camera image as a (H, W, 3) BGR numpy array,
            or None if no image has been received yet.
        """
        return self._image_raw

    @staticmethod
    def _parse_image(weak_self, image):
        """
        Callback that fires every time the sensor produces a new image.

        This runs on a BACKGROUND THREAD managed by the CARLA client.
        We convert the raw CARLA image data into a numpy array and store
        it in self._image_raw for the main thread to read.

        Parameters
        ----------
        weak_self : weakref
            Weak reference to the CameraManager instance.
        image : carla.Image
            The raw image data from the CARLA sensor.
        """
        self = weak_self()
        if not self:
            return

        # Apply the color converter (e.g., CityScapesPalette for segmentation)
        image.convert(self._sensors[self._index][1])

        # Convert the flat byte buffer to a numpy array
        # CARLA images are BGRA (4 channels), we drop the alpha channel
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        array = array[:, :, :3]                                 # BGR only

        # Store a copy (the original buffer is owned by CARLA and may be recycled)
        self._image_raw = array.copy()


# ================================================================================
# WORLD (ties everything together)
# ================================================================================

# Default spawn location: high up to avoid collision on spawn (the vehicle
# will fall to the road surface due to gravity).
START_POSITION = carla.Transform(carla.Location(x=180.0, y=199.0, z=40.0))


def _find_weather_presets():
    """
    Discover all weather presets defined in carla.WeatherParameters.

    CARLA defines presets as class-level attributes with CamelCase names
    (e.g., ClearNoon, WetCloudySunset). We extract them and convert the
    CamelCase to a readable "Clear Noon" format.

    Returns
    -------
    list of (carla.WeatherParameters, str)
        Each entry is a tuple of (preset_object, human_readable_name).
    """
    # Regex to split CamelCase: "ClearNoon" -> ["Clear", "Noon"]
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    to_name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))

    # Find all attributes starting with an uppercase letter
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), to_name(x)) for x in presets]


class World:
    """
    Manages the CARLA simulation world: ego vehicle, sensors, weather.

    On initialization it:
    1. Spawns a Tesla Cybertruck at START_POSITION
    2. Attaches a collision sensor
    3. Creates two CameraManagers:
       - camera_manager          : RGB camera (sensor index 0)
       - camera_manager_semantic : Semantic Segmentation camera (sensor index 5)

    Attributes
    ----------
    world : carla.World
        The CARLA world object.
    hud : HUD
        The HUD/lane-detection processor.
    vehicle : carla.Actor
        The ego vehicle.
    collision_sensor : CollisionSensor
        Detects collisions.
    camera_manager : CameraManager
        RGB camera feed.
    camera_manager_semantic : CameraManager
        Semantic segmentation camera feed.
    """

    def __init__(self, carla_world, hud, config):
        """
        Parameters
        ----------
        carla_world : carla.World
            The world object from carla.Client.get_world().
        hud : HUD
            The HUD instance for display and lane detection.
        config : Config
            Configuration with resolution and network settings.
        """
        self.world = carla_world
        self.hud = hud
        self.config = config

        # Spawn the ego vehicle (Tesla Cybertruck)
        blueprint_library = self.world.get_blueprint_library()
        blueprint = blueprint_library.find('vehicle.tesla.cybertruck')
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = None

        for spawn_point in spawn_points:
            self.vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
            if self.vehicle is not None:
                break

        if self.vehicle is None:
            raise RuntimeError("Could not find a free spawn point")

        # Attach collision sensor
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)

        # Create RGB camera (sensor 0 = 'sensor.camera.rgb' with cc.Raw)
        self.camera_manager = CameraManager(self.vehicle, self.hud, config)
        self.camera_manager.set_sensor(0, notify=False)

        # Create Semantic Segmentation camera
        # (sensor 5 = 'sensor.camera.semantic_segmentation' with CityScapesPalette)
        self.camera_manager_semantic = CameraManager(
            self.vehicle, self.hud, config
        )
        self.camera_manager_semantic.set_sensor(5, notify=False)

        # Weather cycling support
        self._weather_presets = _find_weather_presets()
        self._weather_index = 0

    def restart(self):
        """
        Respawn the vehicle and all sensors at the current position.

        The vehicle is raised 2m to avoid spawning inside the ground,
        and rotation is reset to level. A random Tesla Model 3 is chosen
        as the new vehicle blueprint.
        """
        cam_index = self.camera_manager._index
        cam_pos_index = self.camera_manager._transform_index

        # Get current position and lift slightly
        start_pose = self.vehicle.get_transform()
        start_pose.location.z += 2.0
        start_pose.rotation.roll = 0.0
        start_pose.rotation.pitch = 0.0

        # Pick a random Model 3 blueprint
        blueprint = self._get_random_blueprint()

        # Destroy old actors and respawn
        self.destroy()
        self.vehicle = self.world.spawn_actor(blueprint, start_pose)
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)

        # Recreate RGB camera
        self.camera_manager = CameraManager(
            self.vehicle, self.hud, self.config)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)

        # Update semantic camera transform index
        self.camera_manager_semantic._transform_index = cam_pos_index

        # Notify
        actor_type = ' '.join(
            self.vehicle.type_id.replace('_', '.').title().split('.')[1:])
        if self.hud is not None:
            self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """
        Cycle to the next (or previous) weather preset.

        Parameters
        ----------
        reverse : bool
            If True, cycle backward through the preset list.
        """
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        if self.hud is not None:
            self.hud.notification('Weather: %s' % preset[1])
        self.vehicle.get_world().set_weather(preset[0])

    def tick(self, clock):
        """Update the HUD each simulation tick."""
        if self.hud is not None:
            self.hud.tick(self, clock)

    def render(self, display):
        """
        Retrieve the latest images from both cameras and run lane detection.

        Parameters
        ----------
        display : unused
            Kept for API compatibility.

        Returns
        -------
        key : int
            Keycode from cv2.waitKeyEx().
        raw_image : np.ndarray
            The raw RGB camera image.
        result_image : np.ndarray
            The RGB image with lane detection overlay.
        """
        key = 0
        raw_image = self.camera_manager.render(display)
        semantic_image = self.camera_manager_semantic.render(display)

        result_image = raw_image
        if self.hud is not None:
            self.hud.name = "raw"
            key, result_image = self.hud.render(semantic_image, raw_image)

        return key, raw_image, result_image

    def destroy(self):
        """
        Destroy all spawned actors (vehicle and sensors).

        IMPORTANT: Always call this when your script exits! Actors left
        in the simulation persist even after your Python script terminates.
        """
        actors_to_destroy = [
            self.camera_manager.sensor,
            self.camera_manager_semantic.sensor,
            self.collision_sensor.sensor,
            self.vehicle
        ]
        for actor in actors_to_destroy:
            if actor is not None:
                actor.destroy()

    def _get_random_blueprint(self):
        """
        Get a random Tesla Model 3 blueprint with a random color.

        Returns
        -------
        carla.ActorBlueprint
            A configured vehicle blueprint ready for spawning.
        """
        bp = random.choice(
            self.world.get_blueprint_library().filter('vehicle')
                .filter('model3')
        )
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp


# ================================================================================
# MAIN ENTRY POINT
# ================================================================================
def main():
    """
    Main function: connects to CARLA, spawns the world, and runs the
    main loop until ESC is pressed.

    The main loop:
    1. world.render()  -> retrieves camera images, runs lane detection, displays
    2. controller.parse_events() -> handles keyboard input
    3. Checks for ESC to exit
    4. On exit (or crash), the finally block ensures all actors are destroyed
    """
    config = Config()

    # --- Connect to the CARLA server ---
    print("Connecting to CARLA server at %s:%d..." % (config.host, config.port))
    client = carla.Client(config.host, config.port)
    client.set_timeout(10.0)  # Wait up to 10 seconds for server response
    print("Connected! Server version: %s" % client.get_server_version())

    # --- Create the HUD (lane detection + display manager) ---
    hud = HUD(config.width, config.height)

    # --- Initialize the world ---
    world = None
    try:
        world = World(client.get_world(), hud, config)
        print("Vehicle spawned: %s" % world.vehicle.type_id)

        # Switch both cameras to front-facing view
        world.camera_manager.toggle_camera()
        world.camera_manager_semantic.toggle_camera()

        # Create keyboard controller (starts with autopilot ON by default)
        controller = KeyboardControl(world, config.autopilot)
        print("Autopilot: ON  (press F5 to toggle, ESC to quit)")

        tt = 0
        # Frame counter for periodic saving.

        im_id = 0
        # Saved image counter.

        import os
        if not os.path.exists("images"):
            os.makedirs("images")

        # --- Main loop ---
        while True:
            # 1. Render: get images, run lane detection, display in OpenCV
            key, raw_img, result_img = world.render(None)

            if result_img is not None and tt >= 100:
                im_id += 1
                cv2.imwrite("images/img_{}.png".format(im_id), result_img)
                tt = 0
            # Save the processed image every 100 frames.

            tt += 1
            # Increase frame counter.

            # 2. Handle keyboard input
            controller.parse_events(world, key)

            # 3. Exit on ESC
            if key == 27:
                print("ESC pressed. Exiting...")
                break

    finally:
        # --- Cleanup: ALWAYS destroy actors, even if an exception occurred ---
        if world is not None:
            world.destroy()
            print("All actors destroyed. Goodbye!")


# Run main() when the script is executed directly
if __name__ == '__main__':
    main()
