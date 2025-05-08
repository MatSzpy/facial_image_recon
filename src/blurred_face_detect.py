import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from skimage.feature import canny
from skimage.morphology import opening, closing, square
from skimage.filters import threshold_otsu


class BlurredFaceDetector:
    def __init__(self, data_path):
        self.data_path = data_path

    def rectangles_outlines(self, erode, gray_image):
        # Detect all contours
        contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        rectangles = []

        # Filter rectangles from contours
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            # Specify number of edges
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Detect contours that are rectangles
            if (
                len(approx) == 4  # Check number of edges (rectangle should have 4 edges)
                and (
                    w >= 40 and h >= 40
                )  # Rectangles grater than or equal 40 x 40 pixels
                and (
                    w < gray_image.shape[1] / 2 and h < gray_image.shape[0] / 2
                )  # Rectangles with edges shorter than 1/2 of image width/height
                and (
                    w > gray_image.shape[0] / 20 and h > gray_image.shape[1] / 20
                )  # Rectangles with edges greater than 1/20 of image width/height
                and (
                    x + w < gray_image.shape[1]
                    and h + y < gray_image.shape[0]
                    and x > 0
                    and y > 0
                )  # Rectangles with edges that do not end or start at image edges
                and (
                    ((w > h) and (w < 2 * h)) or ((w < h) and (h < 2 * w))
                )  # Rectangles with balanced proportions (not enlarged)
            ):
                # Rectangle surrounding the blurred area
                rectangle_area = gray_image[y : y + h, x : x + w]

                # Analyze image sharpness using magnitude spectrum
                fft_image = np.fft.fft2(rectangle_area)
                fft_shifted = np.fft.fftshift(fft_image)
                magnitude_spectrum = 20 * np.log(np.abs(fft_shifted))

                # Apply global Otsu thresholding
                thresh_area = threshold_otsu(rectangle_area)
                binary_area = rectangle_area > thresh_area
                binary_area = binary_area.astype(np.uint8) * 255

                # Ignore solid-colored areas with low magnitude spectrum
                if (np.mean(magnitude_spectrum) > 95) and (np.sum(binary_area == 0) / binary_area.size < 0.8 and np.sum(binary_area == 255) / binary_area.size < 0.8):
                    # Add contour to rectangle list
                    rectangles.append((np.mean(magnitude_spectrum), x, y, w, h))

        return rectangles

    def show_stages(self, name_list, image_list):
        # Show stages of detection process
        x = 1
        for image in image_list[:-1]:
            # Plot input image
            plt.subplot(2, 3, x)
            plt.imshow(image)
            plt.title(name_list[x - 1])
            plt.axis("off")
            x += 1
        plt.show()

        # Show final image
        plt.plot(150, 150)
        plt.title(name_list[-1])
        plt.imshow(image_list[-1])
        # plt.show()

    def process_image(self, filename, image_num):
        # Load an input image
        image = cv2.imread(os.path.join(data_path, filename))
        print(filename)

        # Determine size of the image
        height, width, _ = image.shape

        # Create an empty image
        empty_image = np.zeros((height + 1, width + 1, 3), dtype=np.uint8)

        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply adaptive Niblack thresholding
        binary_niblack = cv2.ximgproc.niBlackThreshold(
            gray_image, maxValue=255, type=cv2.THRESH_BINARY, blockSize=3, k=0.8
        )

        # Detect edges on binary image
        edges = canny(
            opening(closing(binary_niblack, square(21)), square(10)), sigma=0.8
        )

        # Connect separated lines
        kernel = np.ones((0, 0), np.uint8)

        # Close huge gaps between lines
        huge_dilate = cv2.dilate((edges.astype(np.uint8)) * 255, kernel, iterations=11)
        huge_erode = cv2.erode(huge_dilate, kernel, iterations=4)

        # Close small gaps between lines
        small_dilate = cv2.dilate((edges.astype(np.uint8)) * 255, kernel, iterations=3)
        small_erode = cv2.erode(small_dilate, kernel, iterations=2)

        # Find rectangles on image
        huge_rectangles = detector.rectangles_outlines(huge_erode, gray_image)
        small_rectangles = detector.rectangles_outlines(small_erode, gray_image)
        rectangles = huge_rectangles + small_rectangles

        # Sort rectangles according to FFT magnitude
        sorted_rectangles = sorted(rectangles, key=lambda item: item[0], reverse=True)

        # Draw rectangles on image
        if sorted_rectangles != []:
            # Draw rectangles on empty image
            for i in sorted_rectangles:
                rectangles_image = cv2.rectangle(
                    empty_image,
                    (i[1], i[2]),
                    (i[1] + i[3], i[2] + i[4]),
                    (0, 255, 0),
                    3,
                )

            # Detect all external contours
            rectangles_image = cv2.cvtColor(rectangles_image, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(
                rectangles_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            face_num = 1

            # Draw found contours on image
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Crop and save image to subfolder
                cropped_image = image[3 + y : y + h - 6, 3 + x : x + w - 6]
                type = filename.rsplit("_", 1)[-1]
                cv2.imwrite(
                    data_path
                    + f"\\blurred_faces\\image{image_num}_face{face_num}_{type}",
                    cropped_image,
                )
                face_num += 1
        else:
            rectangles_image = empty_image

        # Show stages of detection process
        image_list = [
            cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB),
            binary_niblack,
            edges,
            huge_erode,
            small_erode,
            rectangles_image,
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        ]
        name_list = [
            "Input grayscale",
            "Niblack thresholding",
            "Canny edges",
            "Closing huge gaps",
            "Closing small gaps",
            "Found rectangles",
            "Final image",
        ]
        detector.show_stages(name_list, image_list)

    def run(self):
        image_num = 0
        if not os.path.exists(data_path+"\\blurred_faces"):
            os.makedirs(data_path+"\\blurred_faces")
        for filename in os.scandir(data_path):
            start_time = time.time()
            filename = os.path.basename(filename)
            if filename.endswith((".jpg", ".png", ".jpeg")):
                self.process_image(filename, image_num)
            end_time = time.time()
            image_num += 1
            print(f"Execution time: {end_time - start_time}")


if __name__ == "__main__":
    data_path = input(
        "Enter copied path to the folder " \
        "where images with detected faces are stored: "
    )
    detector = BlurredFaceDetector(data_path)
    detector.run()
