# reservoir-pipeline
Pipeline to use reservoir computing to predict variables or forecast the input signal itself


## Installation of reservoirpy
```
git clone git@github.com:ReservoirBrains/reservoir-pipeline.git
cd reservoir-pipeline
```

(Replace the version of python with yours. It works from 3.9 to 3.13)
```
conda create -n rpy-brainhack python=3.13 pip
conda activate rpy-brainhack
pip install -r requirements.txt
```

## Ready-to-use reservoir code
You can directly use the following for automatic optimization and training of your task:
- python script: "test-predict-random-optim-HP.py"
    Read the instruction at the beginning of the file,
    and insert your dataset and training scheme in the "main()" method.
- python notebook: "notebooks/tutorial_hyperopt_reservoirpy_esn.ipynb"
    Follow the instruction in the notebook.