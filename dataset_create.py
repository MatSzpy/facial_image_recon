import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time

class DataCleaner:
    def __init__(self, data_path, num_main_folders=10, num_subfolders=10):
        self.main_directory = main_directory
        self.data_path = data_path
        self.num_main_folders = num_main_folders
        self.num_subfolders = num_subfolders

    def move_data_to_main(self, main_folder_path, subfolder_path):
        os.chdir(subfolder_path)
        os.system(f'for %d in (*) do move "%d" "{main_folder_path}"')
        os.chdir(main_folder_path)
        os.rmdir(subfolder_path)
        print("Data moved successfully!")

    def get_files(self, main_folder_path):
        files = []
        for data in os.scandir(main_folder_path):
            if data.is_file() and data.name not in ["deleted.txt", "preserved.txt"]:
                files.append(data.name)
        return files

    def save_lists(self, file_path, files):
        with open(file_path, mode="w+") as file:
            file.write(", ".join(files))

    def delete_files(self, main_folder_path, deleted_files):
        for file_name in deleted_files:
            os.remove(main_folder_path + "/" + file_name)

    def organize_data(self, folder_index):
        main_folder_path = f"{data_path}\\imdb_{str(folder_index)}"

        # Move data from subfolders to main folder and delete subfolders
        for subfolder_index in range(self.num_subfolders):
            subfolder_name = f"\\{str(folder_index)}{str(subfolder_index)}"
            subfolder_path = main_folder_path + f"\\{str(subfolder_name)}"
            self.move_data_to_main(main_folder_path, subfolder_path)

        # Find data in folder and preserve only random 25% of them
        files = self.get_files(main_folder_path)
        odd_files = files[::2]
        odd_num = round(len(odd_files) / 2)  # Preserve every other file (50% of original dataset)
        preserved_files = random.sample(odd_files, odd_num)  # Preserve random files (25% of original dataset)

        # Save lists of preserved and deleted files
        deleted_files = list(set(files).difference(preserved_files))  # Create list of delted files
        self.save_lists(main_folder_path+ f"/preserved_{folder_index}.txt", preserved_files)
        self.save_lists(main_folder_path+ f"/deleted_{folder_index}.txt", deleted_files)

        # Delete files
        self.delete_files(main_folder_path, deleted_files)

        # Summarize output
        print(f"Total files: {len(files)}")
        print(f"Preserved files: {len(preserved_files)}")
        print(f"Deleted files: {len(deleted_files)}")

        # Move data from subfolders to main folder and delete subfolders
        self.move_data_to_main(self.data_path, main_folder_path)

    def run(self):
        for folder_index in range(self.num_main_folders):
            self.organize_data(folder_index)

class ImageModifier:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(model_path, "", (0, 0), score_threshold=0.8)
        self.main_directory = main_directory

    def adjust_box(self, face):
        box = list(map(int, [max(0, face[0] - face[2] // 20), max(0, face[1] - face[3] // 20), face[2] + face[2] // 10, face[3] + face[3] // 10]))
        return box

    def pixelate_face(self, image, pixels_x, pixels_y):
        pixels_width = np.linspace(0, image.shape[:2][1], pixels_x + 1, dtype="int")
        pixels_height = np.linspace(0, image.shape[:2][0], pixels_y + 1, dtype="int")

        for i in range(1, len(pixels_height)):
            for j in range(1, len(pixels_width)):
                start_width = pixels_width[j - 1]
                start_height = pixels_height[i - 1]
                end_width = pixels_width[j]
                end_height = pixels_height[i]
                resized_image = cv2.resize(image, (pixels_x, pixels_y), interpolation=cv2.INTER_NEAREST_EXACT)
                mosaic = (resized_image[i - 1, j - 1, 0], resized_image[i - 1, j - 1, 1], resized_image[i - 1, j - 1, 2])
                (B, G, R) = [int(x) for x in mosaic]
                cv2.rectangle(image, (start_width, start_height ), (end_width, end_height), (B, G, R), -1)
        return image

    def save_image(self, image, filename, subfolder):
        special_folder = os.path.join(main_directory, subfolder)
        os.makedirs(special_folder, exist_ok=True)
        cv2.imwrite(os.path.join(special_folder, filename), image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def process_image(self, filename):

        image = cv2.imread(os.path.join(data_path, filename))

        height, width, _ = image.shape

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(image)

        modification_type = {"blur5": 5, "blur10": 10, "mosaic10": 10, "mosaic20": 20}
        processed_images = {key: image.copy() for key in modification_type}

        if faces is not None:
            count_faces = 0

            for face in faces:
                box = self.adjust_box(face)

                if box[2] < 40 or box[3] < 40:
                    continue

                count_faces += 1

                for key, param in modification_type.items():
                    if "blur" in key:
                        processed_images[key][box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = cv2.blur(processed_images[key][box[1]:box[1]+box[3],box[0]:box[0]+box[2]], ((box[2]// param),(box[3]// param)))
                    elif "mosaic" in key:
                        pixels_x = param
                        pixels_y = int(box[3] / (box[2] / pixels_x))
                        processed_images[key][box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = self.pixelate_face(processed_images[key][box[1]:box[1]+box[3],box[0]:box[0]+box[2]], pixels_x, pixels_y)

            if count_faces > 0:
                subfolder = f"blurred_{count_faces}" if count_faces < 4 else "blurred_m"

                for key, img in processed_images.items():
                    self.save_image(img, f"{os.path.splitext(filename)[0]}_{key}.jpg", subfolder)


    def run(self):
        for filename in os.scandir(data_path):
            start_time = time.time()
            filename = os.path.basename(filename)
            if filename.endswith(('.jpg', '.png', '.jpeg')) and filename != 'desktop.ini':
                self.process_image(filename)
            end_time = time.time()
            print(f"Execution time: {end_time - start_time} s")

if __name__ == "__main__":
    main_directory = input("Enter copied path to the folder where unzipped IMDB data folder and model are stored: ")
    data_path = os.path.join(main_directory, "data")
    num_main_folders = input("How many main folders?\n")
    num_subfolders = input("How many subfolders?\n")
    cleaner = DataCleaner(data_path, int(num_main_folders), int(num_subfolders))
    cleaner.run()

    model_path = main_directory + "\\face_detection_yunet_2023mar.onnx"
    modifier = ImageModifier(model_path)
    modifier.run()
