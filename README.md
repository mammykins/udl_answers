# udl_answers
[Understanding Deep Learning](https://udlbook.com/) notebooks (with the TODO completed) and chapter problems with answers in notebook form. These notebook files are prefixed with the relevant Chapter and section.

I bought the physical book to work from, recommended!

CITATION:

@book{prince2023understanding,
    author = "Simon J.D. Prince",
    title = "Understanding Deep Learning",
    publisher = "The MIT Press",
    year = 2023,
    url = "http://udlbook.com"}

# Additional python scripts written for my own understanding and practice

I wrote some additional python scripts to help me understand the concepts in the book. I will include them in this repository. They are not part of the book.

## Getting Started

Follow these instructions to set up your development environment and get started with the project.

### Prerequisites

Make sure you have Python and `pip` installed on your system. You can download Python from [python.org](https://www.python.org/).

### Setting Up a Virtual Environment

1. **Install `virtualenv` (if you don't have it already):**
   ```sh
   pip install virtualenv
   ```

2. **Create a virtual environment:**
   Navigate to your project directory and run:
   ```sh
   virtualenv venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```sh
     source venv/bin/activate
     if [[ "$VIRTUAL_ENV" != "" ]]; then echo "Virtualenv is active"; else echo "Virtualenv is not active"; fi
     ```

### Installing Dependencies

1. **Install PyTorch and other dependencies:**
   You can install PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/). For example, to install the CPU version of PyTorch, you can run:
   ```sh
   pip install torch torchvision torchaudio
   ```

2. **Install `numpy`:**
   ```sh
   pip install numpy
   if [[ "$VIRTUAL_ENV" != "" ]]; then echo "Virtualenv is active: $(basename $VIRTUAL_ENV)"; python --version; pip list; else echo "Virtualenv is not active"; fi
   ```

3. **Generate `requirements.txt`:**
   Once all dependencies are installed, you can generate the `requirements.txt` file by running:
   ```sh
   pip freeze > requirements.txt
   ```

### Deactivating the Virtual Environment

When you are done working in the virtual environment, you can deactivate it by running:
```sh
deactivate
```

### Using the `requirements.txt` File

To set up the same environment on another machine, follow these steps:

1. **Clone the repository:**
   ```sh
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create and activate a virtual environment:**
   ```sh
   virtualenv venv
   source venv/bin/activate  # or .\venv\Scripts\activate on Windows
   ```

3. **Install dependencies from `requirements.txt`:**
   ```sh
   pip install -r requirements.txt
   ```

Now you are ready to start working on the project!
