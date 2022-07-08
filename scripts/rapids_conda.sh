CONDA_ENV_NAME="GiDR_DUN_rapids"
CONDA_FILE=$(which conda)

if [ ! -f "$CONDA_FILE" ]; then
    HAS_CONDA=False;
else
    HAS_CONDA=True;
    ENV_DIR="$(conda info --base)";
    MY_ENV_DIR="${ENV_DIR}/envs/${CONDA_ENV_NAME}";
fi

if [ $HAS_CONDA = True ]; then
    if [ -d $MY_ENV_DIR ]; then
        echo ">>> Found ${CONDA_ENV_NAME} environment in ${MY_ENV_DIR}. Skipping installation...";
    else
        echo ">>> Detected conda, but ${CONDA_ENV_NAME} is missing in ${ENV_DIR}. Installing ...";
        echo ">>> (This will take a while)"
        # FIXME -- this should run on user's preferred cuda
        conda clean -a
        conda create -n $CONDA_ENV_NAME \
            -c rapidsai \
            -c nvidia \
            -c conda-forge cuml=22.04 python=3.8 cudatoolkit=11.5 cuda-python=11.7.0;
    fi;
else
    echo ">>> Install conda first.";
fi
