import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class Detector:

    def __init__(self, empty_image: np.ndarray, glass_image: np.ndarray, debug=False):
        self.debug = debug
        self.glass_image = glass_image
        self.empty_image = empty_image
        self.empty_image_copy = empty_image.copy()
        extracted, image_after = Detector.__from_image(self.empty_image, self.glass_image, self.debug)
        self.extracted_glass, self.glass_height, self.glass_box = Detector.__glass_features(extracted, image_after, self.debug)

    def get_level(self, image):
        extracted, image_after = Detector.__from_image(self.empty_image_copy, image)
        img, levels = Detector.__extract_features(extracted, image_after, self.glass_box, self.debug)
        cv.drawContours(img, [self.glass_box], -1, (255, 0, 0), 3)
        Detector.__auto_debug_plot(self.debug, img, 'Result')
        return img, levels

    @staticmethod
    def __from_image(image_before: np.ndarray, image_after: np.ndarray, debug=False):
        image_after = image_after.copy()
        image_before = Detector.__crop_image(image_before, 200, 50)
        image_after = Detector.__crop_image(image_after, 200, 50)
        diff = cv.absdiff(image_before, image_after)
        img_before_thresh, img_before_fixed, img_before = Detector.__normalize_image(image_before, debug)
        img_after_thresh, img_after_fixed, img_after = Detector.__normalize_image(image_after, debug)
        img_diff = cv.absdiff(img_before, img_after)
        diff_thresh, diff_fixed, diff = Detector.__normalize_image(img_diff, debug)
        extracted, mask = Detector.__extract_mask(diff_thresh, diff, debug)
        return extracted, image_after

    @staticmethod
    def __glass_features(mask, image, debug=False):
        gray = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        ret, threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
        largest_contour, _ = Detector.__get_largest_contour(threshold)
        glass_height = 0
        box = None
        if largest_contour is not None:
            rect = cv.minAreaRect(largest_contour)
            Detector.__auto_debug_plot(debug, mask, 'Thresh', None, cmap='gray')
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(image, [box], -1, (255, 0, 0), 3)
            Detector.__auto_debug_plot(debug, image, 'Glass Bounding Box')
            glass_height = Detector.__compute_height(box)
        return image, glass_height, box

    @staticmethod
    def __normalize_image(diff_image, debug=False):
        image = cv.GaussianBlur(diff_image, (7, 7), 0)
        image = Detector.__unsharp_mask(image)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        adaptive_thresh = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 125, 1)
        ret, fixed_thresh = cv.threshold(gray_image, 50, 140, cv.THRESH_BINARY)
        Detector.__auto_debug_plot(debug, image, 'Normalized Image')
        Detector.__auto_debug_plot(debug, gray_image, 'Gray Image', None, cmap='gray')
        Detector.__auto_debug_plot(debug, adaptive_thresh, 'Adaptive Threshold Image', None, cmap='gray')
        Detector.__auto_debug_plot(debug, fixed_thresh, 'Fixed Threshold Image', None, cmap='gray')
        return adaptive_thresh, fixed_thresh, image

    @staticmethod
    def __extract_mask(threshold, image, debug=False):
        contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img_contour = image.copy()
        largest_ratio = largest_perimeter = largest_area = 0
        largest_contour = None
        for contour in contours:
            area = cv.contourArea(contour)
            perimeter = cv.arcLength(contour, True)
            if perimeter > 0:
                ratio = area / perimeter
                if area > largest_area:
                    largest_contour = contour
                    largest_ratio = ratio
                    largest_perimeter = perimeter
                    largest_area = area
        cv.drawContours(img_contour, contours, -1, (0, 255, 0), 1)
        Detector.__auto_debug_plot(debug, img_contour, 'Image Contour')
        epsilon = 0
        edge = cv.approxPolyDP(largest_contour, epsilon, True)
        mask = np.zeros((image.shape[0], image.shape[1]), 'uint8') * 125
        cv.fillConvexPoly(mask, edge, 255, 1)
        extracted = np.zeros_like(image)
        extracted[mask == 255] = image[mask == 255]
        extracted[np.where((extracted == [125, 125, 125]).all(axis=2))] = [0, 0, 20]
        Detector.__auto_debug_plot(debug, extracted, 'Extracted mask', None, cmap='gray')
        return extracted, mask

    @staticmethod
    def __extract_features(mask, image, limit_box, debug=False):
        beer_level = 0
        foam_level = 0
        glass_height = 0
        gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        Detector.__auto_debug_plot(debug, image, 'Beer Box')
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        fluid_min_range = np.array([0, 133, 0])
        fluid_max_range = np.array([12 , 255, 135])
        threshold = cv.inRange(hsv_image, fluid_min_range, fluid_max_range)
        largest_contour, _ = Detector.__get_largest_contour(threshold)
        if largest_contour is not None:
            rect = cv.minAreaRect(largest_contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            if box[1][1] >= limit_box[1][1]:
                cv.drawContours(image, [box], -1, (0, 0, 255), 3)
                Detector.__auto_debug_plot(debug, image, 'Beer Box')
                beer_level = Detector.__compute_height(box)

        foam_min = np.array([0, 48, 148])
        foam_max = np.array([179, 255, 255])
        threshold = cv.inRange(hsv_image, foam_min, foam_max)
        largest_contour, _ = Detector.__get_largest_contour(threshold)
        if largest_contour is not None:
            rect = cv.minAreaRect(largest_contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            if 350 >= box[1][1] >= limit_box[1][1]:
                foam_level = Detector.__compute_height(box)
                cv.drawContours(image, [box], -1, (0, 255, 0), 3)
                Detector.__auto_debug_plot(debug, image, 'Foam Box')
        return image, (beer_level, foam_level)

    @staticmethod
    def __compute_height(box):
        return 0 if box is None else box[3][1] - box[1][1]

    @staticmethod
    def __get_largest_contour(threshold_image):
        contours, hierarchy = cv.findContours(threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        largest_ratio = largest_perimeter = largest_area = 0
        largest_contour = None
        for contour in contours:
            area = cv.contourArea(contour)
            perimeter = cv.arcLength(contour, True)
            if perimeter > 0:
                ratio = area / perimeter
                if area > largest_area:
                    largest_contour = contour
                    largest_ratio = ratio
                    largest_perimeter = perimeter
                    largest_area = area
        return largest_contour, contours

    @staticmethod
    def __crop_image(image, from_width=0, from_height=0):
        width, height, depth = image.shape
        image = image[from_width: width - 170, from_height: height, 0: depth]
        return image

    @staticmethod
    def __unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=50.0, threshold=30):
        blurred = cv.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    @staticmethod
    def __auto_debug_plot(debug, image, title=None, image_color=cv.COLOR_BGR2RGB, **kwargs):
        if debug:
            Detector.__debug_plot(image, title=title, image_color=image_color, **kwargs)

    @staticmethod
    def __debug_plot(image, title, *, image_color=cv.COLOR_BGR2RGB, **kwargs):
        plt.axis('off')
        plt.imshow(image if image_color is None else cv.cvtColor(image, image_color), **kwargs)
        if title:
            plt.title(title)
        plt.show()
