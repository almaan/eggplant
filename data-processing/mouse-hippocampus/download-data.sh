#!/usr/bin/bash

OUT_DIR="../../../data/mouse-hippocampus/raw"
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi

for ii in hippo-slide-seq hippo-visium; do
    if [ ! -d $OUT_DIR/$ii ]; then
        mkdir $OUT_DIR/$ii
    fi
done

echo ------ Downloading Visium data ------
#Visium data
curl https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Adult_Mouse_Brain/V1_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5 --output $OUT_DIR/hippo-visium/filtered_feature_bc_matrix.h5
curl https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Adult_Mouse_Brain/V1_Adult_Mouse_Brain_spatial.tar.gz --output $OUT_DIR/hippo-visium/spatial.tar.gz
tar -xf $OUT_DIR/hippo-visium/spatial.tar.gz -C $OUT_DIR/hippo-visium/
rm $OUT_DIR/hippo-visium/spatial.tar.gz

# Slide-seqV2
echo ------ Slide-seqV2 data ------
echo "Broad's Single Cell Portal does not currently support direct download, but requires you to login via their web interface"
echo "Please visit:"
echo -e "\thttps://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2"
echo "and download the files:"
echo -e "\t-Puck_200115_08.digital_expression.txt.gz"
echo -e "\t-Puck_200115_08_bead_locations.csv"
echo "save these files in $( realpath $OUT_DIR/hippo-slide-seq )"
