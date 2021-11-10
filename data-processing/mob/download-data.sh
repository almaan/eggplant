#!/usr/bin/bash

set -e

OUT_DIR="../../../data/mob/raw/"
FOLDERS=( "images/full" "images/reduced" "counts" "tmats" )

for f in ${FOLDERS[@]}; do
    if [ ! -d $OUT_DIR/$f ]; then
        mkdir -p $OUT_DIR/$f
    fi
done

for ii in {1..4}; do
    curl https://www.spatialresearch.org/wp-content/uploads/2016/07/Rep${ii}_MOB_count_matrix-1.tsv --output $OUT_DIR/counts/Rep_$ii.tsv
    curl https://www.spatialresearch.org/wp-content/uploads/2016/07/HE_Rep${ii}.jpg --output $OUT_DIR/images/full/Rep_$ii.jpg
    if ! (( $ii == 2)); then
        curl https://www.spatialresearch.org/wp-content/uploads/2016/07/Rep${ii}_MOB_transformation.txt --output $OUT_DIR/tmats/Rep_$ii.txt
    else
        curl https://www.spatialresearch.org/wp-content/uploads/2016/07/Rep${ii}_MOB_transformaton.txt --output $OUT_DIR/tmats/Rep_$ii.txt
    fi
done
for ii in {5..12}; do
    curl https://www.spatialresearch.org/wp-content/uploads/2016/07/Rep${ii}_MOB_count_matrix-1.tsv --output $OUT_DIR/counts/Rep_$ii.tsv
    curl https://www.spatialresearch.org/wp-content/uploads/2016/07/HE_Rep${ii}_MOB.jpg --output $OUT_DIR/images/full/Rep_$ii.jpg

    if (( $ii == 9 ))  || (( $ii == 12 )); then
        curl https://www.spatialresearch.org/wp-content/uploads/2016/07/Rep${ii}_MOB_transformaton.txt --output $OUT_DIR/tmats/Rep_$ii.txt
    else
        curl https://www.spatialresearch.org/wp-content/uploads/2016/07/Rep${ii}_MOB_transformation.txt --output $OUT_DIR/tmats/Rep_$ii.txt
    fi
done

for ii in {1..12}; do convert $OUT_DIR/images/full/Rep_$ii.jpg -resize 21% $OUT_DIR/images/reduced/Rep_$ii.jpg; done
