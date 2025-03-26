import numpy as np
from matplotlib import pyplot as plt
import os
import time
import cv2
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.morphology import opening, closing, square


class BlurredFaceDetector:
    def __init__(self, data_path):
        self.data_path = data_path

    def probabilistic_hough_line(self, edges, lines_image):
        lines = probabilistic_hough_line(edges, line_length=5, line_gap=3)
        for line in lines:
            p0, p1 = line
            # Detect only lines with an angle less than 1 degree and longer than 15 pixels
            if (
                (abs(p0[0] - p1[0]) < 1 and (abs(p0[1] - p1[1]) > 15))
                or (abs(p0[1] - p1[1]) < 1)
                and (abs(p0[0] - p1[0]) > 15)
            ):
                cv2.line(lines_image, p0, p1, (255, 255, 255), 1)

        return lines_image

    def rectangles_outlines(self, erode, gray_image, image):
        # Detect all contours
        contours, hierarchy = cv2.findContours(
            erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_num = -1

        # Filter rectangles from contours
        for contour in contours:
            contour_num += 1
            (x, y, w, h) = cv2.boundingRect(contour)

            # Select only rectangles grater than or equal 50 x 50 pixels
            # and smaller than 3/4 of the entire image area
            if (w >= 100 and h >= 100) and (
                w < 3 * gray_image.shape[1] / 4 and h < 3 * gray_image.shape[0] / 4
            ):
                # Make sure the rectangle is not too wide or too high
                if ((w > h) and (w < 2 * h)) or ((w < h) and (h < 2 * w)):
                    # Make sure the area inside rectangle is blurred and is not inside other rectangle
                    rectangle_area = gray_image[y : y + h, x : x + w]
                    fft_image = np.fft.fft2(rectangle_area)
                    fft_shifted = np.fft.fftshift(fft_image)
                    magnitude_spectrum = 20 * np.log(np.abs(fft_shifted))
                    if (
                        np.mean(magnitude_spectrum) > 100
                        and hierarchy[0][contour_num][3] == -1
                    ):
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        print(np.mean(magnitude_spectrum))

    def show_stages(self, name_list, image_list):
        # Show stages of detection process
        x = 1
        for image in image_list:
            # Plot input image
            plt.subplot(2, 3, x)
            plt.imshow(image)
            plt.title(name_list[x - 1])
            plt.axis("off")
            x += 1
        plt.show()

        # Show final image
        plt.plot(150, 150)
        plt.imshow(image)
        plt.show()

    def process_image(self, filename):
        # Load an input image
        image = cv2.imread(os.path.join(data_path, filename))

        # Determine size of the image
        height, width, _ = image.shape

        # Create an empty image
        lines_image = np.zeros((height + 1, width + 1, 3), dtype=np.uint8)

        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Sharpen image
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        sharpened_image = cv2.addWeighted(gray_image, 10, blurred_image, -9, 0)

        # Detect smooth spots on sharpened image
        noise = canny(sharpened_image, sigma=1)

        # Detect edges of smooth spots
        edges = canny(opening(closing(noise, square(15)), square(10)), sigma=1)

        # Detect horizontal and vertical lines
        for i in range(10):
            detector.probabilistic_hough_line(edges, lines_image)

        gray_lines = cv2.cvtColor(lines_image, cv2.COLOR_RGB2GRAY)

        # Connect separated lines
        kernel = np.ones((5, 5), np.uint8)
        dilate = cv2.dilate(gray_lines, kernel, iterations=5)
        erode = cv2.erode(dilate, kernel, iterations=4)

        # Find rectangles on image
        detector.rectangles_outlines(erode, gray_image, image)

        # Show stages of detection process
        image_list = [
            cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB),
            noise,
            edges,
            gray_lines,
            erode,
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        ]
        name_list = [
            "Input grayscale",
            "Smooth spots",
            "Canny edges",
            "Probabilistic Hough",
            "Dilating & Eroding",
            "Final image",
        ]
        detector.show_stages(name_list, image_list)

    def run(self):
        for filename in os.scandir(data_path):
            start_time = time.time()
            filename = os.path.basename(filename)
            if filename.endswith((".jpg", ".png", ".jpeg")):
                self.process_image(filename)
            end_time = time.time()
            print(f"Execution time: {end_time - start_time}")


if __name__ == "__main__":
    data_path = input(
        "Enter copied path to the folder where images with detected faces are stored: "
    )
    detector = BlurredFaceDetector(data_path)
    detector.run()
