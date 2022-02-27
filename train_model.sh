python train_model.py \
    -mode 'rgb' \
    -root '/home/duyaoo/dy/data/WLASL2000/' \
    -save_model "./checkpoints/order_transformer/0905/" \
    -weight "None" \
    -train_split "./preprocess/nslt_100.json" \
    -config_file 'configfiles/asl100.ini' \
    -max_epoch 200