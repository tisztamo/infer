EPOCHS=10
for ((i=1;i<=EPOCHS;i++)); do
    echo $i
    python -m experiments.evaluate.train_siamese --data_dir=/mnt/red/train/humanlike/preprocessed/ --logdir=/mnt/red/train/humanlike/logdir --disable_cp=false    
done
