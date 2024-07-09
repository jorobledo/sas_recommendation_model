pip show zenodo-get
if [ $? == 1 ]; then
    echo "no esta, descargando"
    pip install zenodo_get
else
    echo "zenodo-get found, proceeding with download."
fi
zenodo_get --doi=10.5281/zenodo.10119316