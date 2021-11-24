for dataset in Set5 Set14 B100 Urban100
do
  for scale in 2 3 4
  do
#    echo $dataset imdnx${scale}_2
    python test_IMDN.py --save IMDN_IKS30 --checkpoint experiment/imdnx${scale}_3/my_model_best.pth --upscale_factor $scale --dataset $dataset
  done
done