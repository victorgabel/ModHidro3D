# Tridimensional Hidrodynamic Model

This model is the result of an exercise completed for the Numerical Modelling course taught by Prof. Dr. Joseph Harari at the University of São Paulo.

Originally, this model was intended to calculate the effects of wind action in the region of Praia Grande (State of São Paulo, Brazil). I have adapted it to work in the Gulf of Eilat.

## 1. How to run it

### 1.1. Grid
The file `GridTools.ipynb` can generate a valid model grid (in NetCDF format) from a GEBCO topography file. `grid_name` controls the output file name, `author` and `source` changes the metadata. `new_dx` and `new_dy` controls the interpolation data. There is one plot inside the file, aiming to show the results of the grid before running the model.

### 1.2. Model Parameters
Almost all model parameters are controlled in `config.ini`. The `[ENVIRONMENTAL]` section controlls all environmental parameters, such as gravity, diffusion coefficients, water and air density, etc. The `[EXECUTION]` section controlls model running parameters that are independent from the grid, such as $\Delta t$, $\Delta z$, and the maximum number of iterations. The `experiment` parameter sets the folder of where all the files will be saved.

### 1.3. Running
With everything set up, we can look at `run.py`. This three-line file contains the running instructions for Python:
```
from model import Model

model = Model(zplot_1=50, zplot_2=700)
model.run()
```

`zplot_1` and `zplot_2` are plot parameters for the model. These plots are used only to check whether the results are viable or not. 

After executing this file, a progress bar will appear in the terminal. You can check both the pictures and fields as soon as they are saved in their respective folders. Pictures are in PNG format, while fields are in NetCDF format.

## 2. Model Caracteristics

This model considers a hydrostatic homogeneous ocean, with constant wind throughout the execution. All equations were discretized using the centralized finite differencing method.

The used equations where

$$
\begin{dcases}
\dfrac{\partial \eta}{\partial t} + H\left(\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y}\right) = 0\\~\\
\dfrac{\partial u}{\partial t} + u\dfrac{\partial u}{\partial x} + v\dfrac{\partial u}{\partial y} -fv = -g\dfrac{\partial \eta}{\partial x} + \nu_h\left(\dfrac{\partial^2 u}{\partial x^2} + \dfrac{\partial^2 u}{\partial y^2}\right) + \nu_z\dfrac{\partial^2 u}{\partial z^2} - ru\\~\\
\dfrac{\partial v}{\partial t} + u\dfrac{\partial v}{\partial x} + v\dfrac{\partial v}{\partial y} +fu = -g\dfrac{\partial \eta}{\partial y} + \nu_h\left(\dfrac{\partial^2 v}{\partial x^2} + \dfrac{\partial^2 v}{\partial y^2}\right) + \nu_z\dfrac{\partial^2 v}{\partial z^2} - rv
\end{dcases}
$$

## 3. Output
The main model output is located in `{experiment_name}/fields`. The `freqplot` parameter in `config.ini` sets the interval at which fields will be saved.
