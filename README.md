# Neural Network with SMOTE for Generator Unit Classification
#### Author: Bryan Ambrósio

This project applies an Artificial Neural Network (ANN) combined with SMOTE balancing techniques to classify the minimum number of Generator Units (UGs) to be disconnected, based on data extracted from simulations and stability analysis of power systems.

---

## Project Structure

- `main.py` — Main, well-commented Python script containing all data processing, model training, and visualization logic.
- `setup.ann.env.bat` — Windows batch script to automate the creation of a virtual environment and installation of dependencies.
- `requirements.txt` — List of all required Python libraries for the project.
- `Results/` — Output folder where all results (reports, figures, etc.) will be saved after running the code.

---

## How to Run the Project

### Environment Setup (Windows)

To guarantee all dependencies are installed and the correct Python version is used, **always start by running the provided batch script**:

```cmd
setup.ann.env.bat
