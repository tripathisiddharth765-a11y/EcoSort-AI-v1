from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

base_dataset_path = project_root / "Dataset" / "waste-segregation" / "versions" / "6" / "archive_waste" / "archive_waste"

train_path = base_dataset_path / "train"
test_path = base_dataset_path / "test"

# class_paths = {}

# for item in train_path.iterdir():
#     class_paths[item.name] = item

# for class_name,folder in class_paths.items():
#      total_images = sum(1 for item in folder.glob("*") if item.is_file())
#      print(f"{class_name} : {total_images}")    

# print("-"*50)

# for item in test_path.iterdir():
#     class_paths[item.name] = item

# for class_name,folder in class_paths.items():
#      total_images = sum(1 for item in folder.glob("*") if item.is_file())
#      print(f"{class_name} : {total_images}")    
      















