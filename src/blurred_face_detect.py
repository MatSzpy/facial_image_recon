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

    def save_list(self, file_path, files):
        with open(file_path, mode="a+") as file:
            file.write(", ".join(files) + "\n")

    def rectangles_outlines(self, erode, gray_image):
        # Detect all contours
        contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        rectangles = []

        # Filter rectangles from contours
        for contour in contours:
            # Specify number of edges
            epsilon = 0.06 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            (x, y, w, h) = cv2.boundingRect(approx)

            image_x = gray_image.shape[1]
            image_y = gray_image.shape[0]

            contour_area = cv2.contourArea(approx)
            coverage = contour_area / (w * h) if w * h > 0 else 0

            # Detect contours that are rectangles
            if (
                len(approx) == 4  # Check number of edges (rectangle should have 4 edges)
                and (
                    w >= 40 and h >= 40
                )  # Rectangles grater than or equal 40 x 40 pixels
                and (
                    w > image_y / 20 and h > image_x / 20
                )  # Rectangles with edges greater than 1/20 of image width/height
                and (
                    x + w < image_x - 15
                    and h + y < image_y - 15
                    and x > 15
                    and y > 15
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

                num_black_px = np.sum(binary_area == 0)
                num_white_px = np.sum(binary_area == 255)

                corners_x = [approx[i][0][0] for i in range(4)]
                corners_y = [approx[i][0][1] for i in range(4)]
                corners = [corners_x, corners_y]

                counter = 0
                for lst in corners:
                    for i in lst:
                        lst = lst[1:]
                        for j in lst:
                            if abs(i - j) < 6:
                                counter += 1

                if (np.mean(magnitude_spectrum) >= 95 and (num_black_px / binary_area.size <= 0.8 and num_white_px / binary_area.size <= 0.8) # Ignore solid-colored areas with low magnitude spectrum
                        and coverage >= 0.6 # Object covers more more than 60% of the area
                        and (cv2.minAreaRect(approx)[2] >= 80  or cv2.minAreaRect(approx)[2] <= 15) # Check if the rotation angle of the minimum area rectangle is near horizontal or vertical
                ):
                    # Add contour to rectangle list
                    if counter > 0:
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
        print("------------------------------------------------")

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

        kernel = np.ones((3, 3), np.uint8)

        # Detect edges on binary image
        binary_niblack_clean = opening(
            closing(binary_niblack, square(21)),
            square(10)
        )
        dilated = cv2.dilate(binary_niblack_clean, kernel)
        eroded = cv2.erode(binary_niblack_clean, kernel)
        edges = cv2.subtract(dilated, eroded)

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
            contours, _ = cv2.findContours(rectangles_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            face_num = 1

            # Draw found contours on image
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Crop and save image to subfolder
                coords = [2 + x, 2 + y, x + w - 4, y + h - 4]
                cropped_image = image[coords[1]:coords[3], coords[0]:coords[2]]
                original_filename, type = filename.rsplit("_", 1)
                cropped_name = f"image{image_num}_face{face_num}_{type}"
                coords = list(map(str, coords))
                cv2.imwrite(data_path + f"\\blurred_faces_final\\{cropped_name}",cropped_image,)
                self.save_list(data_path + f"\\blurred_faces_final\\cropped_images.txt",[original_filename + ".jpg", cropped_name] + coords)
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
            "Edges",
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