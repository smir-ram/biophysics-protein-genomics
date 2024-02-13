"""
Generate `tutorial_name` with placeholder
> python generate_tutorial.py Tutorial_Name

Tutorial_Name/
│
├── Concepts/
│   ├── Introduction.md          # Overview of the tutorial
│   ├── Chapter1_Concept1.md     # Detailed explanation of concept 1
│   ├── Chapter2_Concept2.md     # Detailed explanation of concept 2
│   └── ...
│
├── Code/
│   ├── Source_Code/             # Folder for source code files
│   │   ├── main.py              # Main code file
│   │   ├── utils.py             # Utility functions
│   │   └── ...
│   │
│   ├── Examples/                # Folder for code examples
│   │   ├── example1.py          # Code example 1
│   │   ├── example2.py          # Code example 2
│   │   └── ...
│   │
│   └── README.md                # Instructions for running code, dependencies, etc.
│
├── Docs/
│   ├── Tutorial_Documentation.md # Documentation for the tutorial
│   ├── API_Documentation.md      # Documentation for any APIs used
│   └── ...
│
├── Assets/                      # Folder for images, diagrams, etc.
│   ├── diagrams/                # Diagrams illustrating concepts
│   ├── screenshots/             # Screenshots of code output or examples
│   └── ...
│
├── README.md                    # Overview of the tutorial, setup instructions, etc.
└── LICENSE                      # License information for the tutorial

"""

import os
import sys

def generate_tutorial_structure(tutorial_name):
    # Create the main tutorial folder
    os.makedirs(tutorial_name)

    # Create Concepts folder and files
    concepts_path = os.path.join(tutorial_name, "Concepts")
    os.makedirs(concepts_path)
    open(os.path.join(concepts_path, "Introduction.md"), 'a').close()
    open(os.path.join(concepts_path, "Chapter1_Concept1.md"), 'a').close()
    open(os.path.join(concepts_path, "Chapter2_Concept2.md"), 'a').close()

    # Create Code folder and files
    code_path = os.path.join(tutorial_name, "Code")
    os.makedirs(code_path)
    src_code = tutorial_name.lower().replace(" ","_")
    os.makedirs(os.path.join(code_path, src_code))
    open(os.path.join(code_path, src_code, "main.py"), 'a').close()
    open(os.path.join(code_path, src_code, "utils.py"), 'a').close()
    os.makedirs(os.path.join(code_path, "Examples"))
    open(os.path.join(code_path, "Examples", "example1.py"), 'a').close()
    open(os.path.join(code_path, "Examples", "example2.py"), 'a').close()
    with open(os.path.join(code_path, "README.md"), 'w') as readme:
        readme.write("# Code\n\nInstructions for running code, dependencies, etc.")

    # Create Docs folder and files
    docs_path = os.path.join(tutorial_name, "Docs")
    os.makedirs(docs_path)
    open(os.path.join(docs_path, "Tutorial_Documentation.md"), 'a').close()
    open(os.path.join(docs_path, "API_Documentation.md"), 'a').close()

    # Create Assets folder and subfolders
    assets_path = os.path.join(tutorial_name, "Assets")
    os.makedirs(assets_path)
    os.makedirs(os.path.join(assets_path, "diagrams"))
    os.makedirs(os.path.join(assets_path, "screenshots"))

    # Create README.md and LICENSE files
    with open(os.path.join(tutorial_name, "README.md"), 'w') as readme:
        readme.write(f"# {tutorial_name}\n\nOverview of the tutorial, setup instructions, etc.")
    open(os.path.join(tutorial_name, "LICENSE"), 'a').close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_tutorial.py <tutorial_name>")
        sys.exit(1)
    tutorial_name = sys.argv[1]
    generate_tutorial_structure(tutorial_name)
    print(f"{tutorial_name} structure generated successfully!")
