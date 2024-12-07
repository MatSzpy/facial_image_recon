import os
import random

class DataCleaner:
    def __init__(self, data_path, num_main_folders=10, num_subfolders=10):
        self.data_path = data_path
        self.num_main_folders = num_main_folders
        self.num_subfolders = num_subfolders

    def move_data_to_main(self, main_folder_path, subfolder_name):
        subfolder_path = main_folder_path + f"\\{str(subfolder_name)}"
        os.chdir(subfolder_path)
        os.system(f'for %d in (*) do move "%d" "{main_folder_path}"')
        os.chdir(main_folder_path)
        os.rmdir(subfolder_path)
        print(f"Data from {subfolder_name} moved successfully!")

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
            self.move_data_to_main(main_folder_path, subfolder_name)

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

    def run(self):
        for folder_index in range(self.num_main_folders):
            self.organize_data(folder_index)

if __name__ == "__main__":
    data_path = input("Enter copied path to the folder where unzipped IMDB data folders are stored: ")
    reader = DataCleaner(data_path)
    reader.run()
