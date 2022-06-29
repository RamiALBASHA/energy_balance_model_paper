# energy_balance_model_paper
Project of the paper that describes the energy balance model


Installation
------------
    conda create -n ebp python=3.9
    conda activate ebp
    conda install -c openalea3 -c conda-forge alinea.astk alinea.caribu
    conda install -c conda-forge python-graphviz scikit-learn
    conda install -c itk convert_units
    cd ~/crop_energy_balance
    pip install -e .
    cd ~/crop_irradiance
    pip install -e .