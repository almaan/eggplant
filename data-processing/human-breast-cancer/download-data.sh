#!/usr/bin/bash

OUT_DIR="../../../data/human-breast-cancer/raw"

for ii in bcA bcB; do
    if [ ! -d $OUT_DIR/$ii ]; then
        mkdir -p  $OUT_DIR/$ii
    fi
done

echo ------ Downloading Visium data ------
curl https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5 --output $OUT_DIR/bcA/filtered_feature_bc_matrix.h5
curl https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_spatial.tar.gz  --output $OUT_DIR/bcA/spatial.tar.gz
curl https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Breast_Cancer_Block_A_Section_2/V1_Breast_Cancer_Block_A_Section_2_filtered_feature_bc_matrix.h5 --output $OUT_DIR/bcB/filtered_feature_bc_matrix.h5
curl https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Breast_Cancer_Block_A_Section_2/V1_Breast_Cancer_Block_A_Section_2_spatial.tar.gz  --output $OUT_DIR/bcB/spatial.tar.gz

for sample in bcA bcB; do
    tar -xf $OUT_DIR/$sample/spatial.tar.gz -C $OUT_DIR/$sample/
    rm $OUT_DIR/$sample/spatial.tar.gz
done
