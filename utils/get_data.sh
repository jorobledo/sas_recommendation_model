pip show zenodo-get
if [ $? == 1 ]; then
    echo "zenodo-get not found. Proceeding to install."
    pip install zenodo_get
else
    echo "zenodo-get found. Proceeding with download."
fi
zenodo_get --doi=10.5281/zenodo.10119316 --output-dir="./data"