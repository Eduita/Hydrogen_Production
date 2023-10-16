All of the files here should allow you to run the model.

The 'Wind data' file contains the optimization results, which are references in:
BASELINE_PPA_models.py
BASELINE_SEP_12_23_deterministic.py
BASELINE_SEP_30_23_monthly.py

---

_instantiate_inputs.py is an object that converts the json files into python dictionaries. json files below
input_parameters_deterministic.json
input_parameters.json

BASELINE_PPA_models is a levelized cost model while BASELINE_SEP_12_23_deterministic.py and BASELINE_SEP_30_23_monthly.py are NPV models. 

BASELINE_global_variables.py contains some important variables constant across all py files. 

--

_GBM.py stands for general brownian motion and is used to predict ammonia, electricity, and natural gas prices

--
Notation:

If a function has a "_" underscore behind it, it means its a helper function. This helper function is never called outside of the object. It's used to split up complex calculations into chunks.

--

These files run the models:

BASELINE_PPA_models.py
BASELINE_SEP_12_23_deterministic.py
BASELINE_SEP_30_23_monthly.py

The other files: 
_MI_OPEX.py
_DCF_Model.py
_instantiate_inputs.py
_CAPEX.py
_CI_Calculator.py
_TAX_CREDITS.py

Are helper objects to calculate the NPV. The dependencies and modules match up to the first figure in the methodology.

This is the general structure:
_CAPEX.py -> _DCF_Model.py
_MI_OPEX.py -> _DCF_Model.py
_CI_Calculator.py -> _TAX_CREDITS.py -> _DCF_Model.py

Then, _DCF_Model.py is called with the output of _instantiate_inputs.py, which instantiates the random inputs into a sample. 

All that BASELINE_PPA_models.py, BASELINE_SEP_12_23_deterministic.py, and BASELINE_SEP_30_23_monthly.py files do is run a loop across simulations, scenarios, and times. In this loop, _instantiate_inputs.py and _DCF_Model.py is called. 

The output is excel files with all of the data. 

The data is then plotted using matplotlib. 

I include a folder called "Visuals" that shows how the data is plotted. There is also some samples data. 



