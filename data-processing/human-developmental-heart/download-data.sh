DATA_DIR="../../data/human-developmental-heart/curated"

if [ ! -d $DATA_DIR ]; then
    mkdir -p $DATA_DIR
fi

echo "The raw sequencing data will be provided upon publication of the manuscript"
echo "Until publication only the curated files will be provided."
echo "These files will be downloaded from the Zenodo repository"

curl XXX --output $DATA_DIR
