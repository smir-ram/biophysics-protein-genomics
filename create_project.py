"""
python create_project.py <project-name>
"""
import os
import sys

def create_project_structure(root_dir):
    # Create project root directory
    os.makedirs(root_dir, exist_ok=True)

    # Define directory structure
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'utils',
        'images',
        'config',
        'notebooks',
        'experiments/experiment_1/logs',
        'experiments/experiment_1/saved_models',
    ]

    files = [
        'data/dataset.py',
        'models/architecture.py',
        'models/loss.py',
        'models/metrics.py',
        'models/train.py',
        'models/predict.py',
        'utils/helpers.py',
        'utils/visualization.py',
        'config/config.yaml',
        'requirements.txt',
        'paper.md',
    ]

    # Create directories
    for directory in directories:
        os.makedirs(os.path.join(root_dir, directory), exist_ok=True)

    # Create empty placeholder files
    for file in files:
        open(os.path.join(root_dir, file), 'a').close()
        
def main():
    project_root_directories  = sys.argv[1:] #'CondonCraft'
    for project_root_directory in project_root_directories:
        create_project_structure(project_root_directory)
        print(f"\nProject structure created at: {project_root_directory}")

if __name__ == "__main__":
    main()
    print ("\nCompleted.")
    
