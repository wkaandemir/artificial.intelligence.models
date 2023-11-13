import os
import subprocess

subprocess.run(["pip3", "install", "-r", "requirements.txt"])

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# List all files in the folder
file_list = [file for file in os.listdir() if file.endswith(".py")]

# Sort the files
file_list.sort()

# Define the file to exclude
exclude_file = "hyperparameters.py"

# Run each file sequentially
for file in file_list:
    # Get the absolute path of the current file
    file_path = os.path.abspath(file)

    if file_path != current_script_path and file != exclude_file:  # Avoid running this script and the excluded file
        print(f"Running {file} file...")
        try:
            # Import the module dynamically
            module = __import__(file[:-3])  # Remove the '.py' extension
            # Call the run function in the module
            module.run()
        except Exception as e:
            print(f"Error in {file}: {e}")

print("Process completed. Exiting...")
