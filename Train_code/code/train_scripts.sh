# # RCAN G10R20 x234
# nohup python main.py --whichgpu 0 --save dml3002 --sparseMode pw --t0 -12 --targetIKSSparsity 60 --decayDistance 10 --k 0.005 --model rcan --pre_train ../experiment/Model_paper/RCAN_BIX2.pt --data_test Set5 --epochs 500 --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 2 --patch_size 96 &
# nohup python main.py --whichgpu 1 --save dml3003 --sparseMode pw --t0 -12 --targetIKSSparsity 60 --decayDistance 10 --k 0.005 --model rcan --pre_train ../experiment/Model_paper/RCAN_BIX3.pt --data_test Set5 --epochs 500 --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 3 --patch_size 144 &
# nohup python main.py --whichgpu 2 --save dml3001 --sparseMode pw --t0 -12 --targetIKSSparsity 60 --decayDistance 10 --k 0.005 --model rcan --pre_train ../experiment/Model_paper/RCAN_BIX4.pt --data_test Set5 --epochs 500 --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 4 --patch_size 192 &

# # RCAN G5R10 x234
# nohup python main.py --whichgpu 0 --save dml6002 --sparseMode pw --t0 -12 --targetIKSSparsity 87   --decayDistance 5 --k 0.005 --model rcan --pre_train ../experiment/RCAN1000nx2/model_best.pt     --data_test Set5 --epochs 500 --lr_decay 200 --reset --chop --n_resgroups 5 --n_resblocks 10 --n_feats 64 --scale 2 --patch_size 96 &
# nohup python main.py --whichgpu 1 --save dml6003 --sparseMode pw --t0 -12 --targetIKSSparsity 91.5 --decayDistance 5 --k 0.005 --model rcan --pre_train ../experiment/RCAN1000nx3/model_best.pt     --data_test Set5 --epochs 500 --lr_decay 200 --reset --chop --n_resgroups 5 --n_resblocks 10 --n_feats 64 --scale 3 --patch_size 144 &
# nohup python main.py --whichgpu 2 --save dml6004 --sparseMode pw --t0 -12 --targetIKSSparsity 90.3 --decayDistance 5 --k 0.005 --model rcan --pre_train ../experiment/RCAN1000n/model/model_best.pt --data_test Set5 --epochs 500 --lr_decay 200 --reset --chop --n_resgroups 5 --n_resblocks 10 --n_feats 64 --scale 4 --patch_size 192 &
#
# # EDSR x234
# nohup python main.py --whichgpu 0 --save dml4007 --sparseMode pw --t0 -12 --targetIKSSparsity 85 --decayDistance 10 --k 0.005 --template EDSR_paper --pre_train ../experiment/Model_paper/EDSR_x2.pt --data_test Set5  --epochs 500 --lr_decay 200 --scale 2 --patch_size 96 --reset &
# nohup python main.py --whichgpu 1 --save dml4006 --sparseMode pw --t0 -12 --targetIKSSparsity 85 --decayDistance 10 --k 0.005 --template EDSR_paper --pre_train ../experiment/Model_paper/EDSR_x3.pt --data_test Set5  --epochs 500 --lr_decay 200 --scale 3 --patch_size 144 --reset &
# nohup python main.py --whichgpu 2 --save dml4005 --sparseMode pw --t0 -12 --targetIKSSparsity 85 --decayDistance 10 --k 0.005 --template EDSR_paper --pre_train ../experiment/Model_paper/EDSR_x4.pt --data_test Set5  --epochs 500 --lr_decay 200 --scale 4 --patch_size 192 --reset &

# # use tensorboard to check intermedia results
# tensorboard --logdir ./experiment/runs

# SAN
#python main.py --whichgpu 0 --save san1001 --model SAN2 --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 200 --lr_decay 200 --reset --chop --n_resgroups 20 --n_resblocks 10 --n_feats 64 --scale 4 --patch_size 192 --pre_train ../../experiment/SAN_pretrain/SAN_BI4X.pt
#python main.py --whichgpu 0 --save san1002 --model SAN2 --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 500 --lr_decay 200 --reset --chop --n_resgroups 20 --n_resblocks 10 --n_feats 64 --scale 4 --patch_size 192 --pre_train ../../experiment/SAN_pretrain/SAN_BI4X.pt
#python main.py --whichgpu 0 --save san1003 --model SAN2 --sparseMode pw --t0 -12 --targetIKSSparsity 60 --decayDistance 10 --k 0.005 --epochs 200 --lr_decay 200 --reset --chop --n_resgroups 20 --n_resblocks 10 --n_feats 64 --scale 4 --patch_size 192 --pre_train ../../experiment/san1002/model/model_latest.pt



# RCAN
python main.py --whichgpu 2 --save dmltest --sparseMode rcan --t0 -12 --targetIKSSparsity 60 --decayDistance 10 --k 0.005 --model rcan --pre_train ../experiment/RCAN_pretrain/RCAN_BIX4.pt --data_test Set5 --epochs 500 --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 4 --patch_size 192
# EDSR
python main.py --whichgpu 0 --save dmltest --sparseMode edsr --t0 -12 --targetIKSSparsity 85 --decayDistance 10 --k 0.005 --template EDSR_paper --pre_train ../experiment/EDSR_pretrain/EDSR_x4.pt --data_test Set5  --epochs 500 --lr_decay 200 --scale 4 --patch_size 192 --reset

# IKS2 TESR PSNR

python main.py --whichgpu 2 --save dmltest --sparseMode pw --template EDSR_paper --pre_train /home/ywhuang/DML/IKS/experiment/dml4007/model/model_best.pt --scale 2 --patch_size 96  --reset --data_test Set5+Set14+B100+Urban100
python main.py --whichgpu 2 --save dmltest --sparseMode pw --template EDSR_paper --pre_train /home/ywhuang/DML/IKS/experiment/dml4006/model/model_best.pt --scale 3 --patch_size 144 --reset --data_test Set5+Set14+B100+Urban100
python main.py --whichgpu 2 --save dmltest --sparseMode pw --template EDSR_paper --pre_train /home/ywhuang/DML/IKS/experiment/dml4005/model/model_best.pt --scale 4 --patch_size 192 --reset --data_test Set5+Set14+B100+Urban100

python main.py --whichgpu 1 --save dmltest --model NLSN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005  --epochs 200  --lr_decay 200  --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 4 --patch_size 192 --pre_train /home/ywhuang/DML/IKS/experiment/pretrain/NLSN_pretrain/NLSN_BIX4.pt --test_only --data_test Set5+Set14+B100+Urban100
python main.py --whichgpu 1 --save dmltest --model NLSN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005  --epochs 200  --lr_decay 200  --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 3 --patch_size 144 --pre_train /home/ywhuang/DML/IKS/experiment/pretrain/NLSN_pretrain/NLSN_BIX3.pt --test_only --data_test Set5+Set14+B100+Urban100
python main.py --whichgpu 1 --save dmltest --model NLSN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005  --epochs 200  --lr_decay 200  --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 2 --patch_size 96  --pre_train /home/ywhuang/DML/IKS/experiment/pretrain/NLSN_pretrain/NLSN_BIX2.pt --test_only --data_test Set5+Set14+B100+Urban100

python main.py --whichgpu 1 --save dmltest --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 500  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 4 --patch_size 192 --pre_train /home/ywhuang/DML/IKS/experiment/pretrain/HAN_pretrain/HAN_BIX4.pt --test_only --data_test Set5+Set14+B100+Urban100
python main.py --whichgpu 1 --save dmltest --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 500  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 3 --patch_size 144 --pre_train /home/ywhuang/DML/IKS/experiment/pretrain/HAN_pretrain/HAN_BIX3.pt --test_only --data_test Set5+Set14+B100+Urban100
python main.py --whichgpu 1 --save dmltest --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 500  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 2 --patch_size 96  --pre_train /home/ywhuang/DML/IKS/experiment/pretrain/HAN_pretrain/HAN_BIX2.pt --test_only --data_test Set5+Set14+B100+Urban100

python main.py --whichgpu 1 --save dmltest --model NLSN --sparseMode pw --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 4 --patch_size 192 --pre_train /home/ywhuang/DML/IKS/experiment/nlsn1002/model/model_best_-1.pt --test_only --data_test Set5+Set14+B100+Urban100 --save_result
python main.py --whichgpu 2 --save dmltest --model NLSN --sparseMode pw --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 3 --patch_size 144 --pre_train /home/ywhuang/DML/IKS/experiment/nlsn1003_f1/model/model_best_-1.pt --test_only --data_test Set5+Set14+B100+Urban100 --save_result
python main.py --whichgpu 2 --save dmltest --model NLSN --sparseMode pw --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 2 --patch_size 96  --pre_train /home/ywhuang/DML/IKS/experiment/nlsn1004_f2/model/model_best_-1.pt --test_only --data_test Set5+Set14+B100+Urban100 --save_result

python main.py --whichgpu 2 --save dmltest --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 500  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 3 --patch_size 144 --pre_train /home/ywhuang/DML/IKS/experiment/han1002_f2/model/model_best_-1.pt --test_only --data_test Set5+Set14+B100+Urban100 --save_result
python main.py --whichgpu 2 --save dmltest --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 500  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 2 --patch_size 96  --pre_train /home/ywhuang/DML/IKS/experiment/han1003_f2/model/model_best_-1.pt --test_only --data_test Set5+Set14+B100+Urban100 --save_result
python main.py --whichgpu 2 --save dmltest --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 500  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 4 --patch_size 192 --pre_train /home/ywhuang/DML/IKS/experiment/han1001_f2/model/model_best_-1.pt --test_only --data_test Set5+Set14+B100+Urban100 --save_result


python main.py --pre_train /home/ywhuang/DML/IKS/experiment/dml3002/model/model_best_-1.pt --whichgpu 1 --save dmltest --sparseMode pw --t0 -12 --targetIKSSparsity 60 --decayDistance 10 --k 0.005 --model rcan --epochs 500 --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 2 --patch_size 96 --test_only --data_test Set5

python main.py --pre_train /home/ywhuang/DML/IKS/experiment/trans_model/rcanx3_iks_trans.pt --whichgpu 1 --save dmltest --sparseMode rcan --t0 -12 --targetIKSSparsity 60 --decayDistance 10 --k 0.005 --model rcan --epochs 500 --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 3 --patch_size 144 --test_only --data_test Set5
python main.py --pre_train /home/ywhuang/DML/IKS/experiment/trans_model/s_rcanx3_iks_trans.pt --whichgpu 1 --save dmltest --sparseMode rcan --model rcan --reset --chop --n_resgroups 5 --n_resblocks 10 --n_feats 64 --scale 3 --patch_size 144 --test_only --data_test Set5

python main.py --whichgpu 2 --save dmltest --sparseMode edsr --template EDSR_paper --pre_train /home/ywhuang/DML/IKS/experiment/trans_model/s_edsrx2_iks_trans.pt --scale 2 --patch_size 96 --reset --test_only --data_test Set5+Set14+B100+Urban100 --save_result

python main.py --whichgpu 2 --save dmltest --model NLSN --sparseMode pw --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 4 --patch_size 192  --pre_train /home/ywhuang/DML/IKS/experiment/trans_model/nlsnx4_iks_trans.pt --test_only --data_test Set5+Set14+B100+Urban100

python main.py --whichgpu 2 --save dmltest --model HAN --sparseMode han --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 500  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 4 --patch_size 192 --pre_train /home/ywhuang/DML/IKS/experiment/trans_model/hanx4_iks_trans.pt --test_only --data_test Set5+Set14


# TEST Parameters and Multi-adds
python dml.py --cpu --save dmltest --sparseMode rcan --pre_train /home/ywhuang/DML/IKS/experiment/pretrain/RCANG5R10_pretrain/RCAN1000nx2/model_best.pt --model rcan --data_test Set5 --reset --n_resgroups 5 --n_resblocks 10 --n_feats 64 --scale 2 --patch_size 96
python dml.py --cpu --save dmltest --sparseMode rcan --pre_train /home/ywhuang/DML/IKS/experiment/trans_model/s_rcanx2_iks_trans.pt                     --model rcan --data_test Set5 --reset --n_resgroups 5 --n_resblocks 10 --n_feats 64 --scale 2 --patch_size 96

python dml.py --cpu --save dmltest --sparseMode edsr --template EDSR_paper --pre_train /home/ywhuang/DML/IKS/experiment/trans_model/s_edsrx2_iks_trans.pt --scale 2 --patch_size 96

python dml.py --cpu --save dmltest --model NLSN --sparseMode nlsn  --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train /home/ywhuang/DML/IKS/experiment/pretrain/NLSN_pretrain/NLSN_BIX2.pt  --scale 2
python dml.py --cpu --save dmltest --model NLSN --sparseMode nlsn  --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train /home/ywhuang/DML/IKS/experiment/trans_model/nlsnx2_iks_trans.pt  --scale 2

python dml.py --cpu --save dmltest --model HAN --sparseMode han --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train /home/ywhuang/DML/IKS/experiment/pretrain/HAN_pretrain/HAN_BIX4.pt --scale 4
python dml.py --cpu --save dmltest --model HAN --sparseMode han --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train /home/ywhuang/DML/IKS/experiment/trans_model/hanx2_iks_trans.pt --scale 2


# model transfer: IKS model -> sparse model
python dml_model_trans.py --cpu --save dmltest --sparseMode pw --t0 -12 --targetIKSSparsity 60 --decayDistance 10 --k 0.005 --model rcan --data_test Set5 --epochs 500 --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 4 --patch_size 192 --pre_train /home/ywhuang/DML/IKS/experiment/dml3001/model/model_best_-1.pt
python dml_model_trans.py --cpu --save dmltest --sparseMode pw --t0 -12 --targetIKSSparsity 60 --decayDistance 10 --k 0.005 --model rcan --data_test Set5 --epochs 500 --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 2 --patch_size 96 --pre_train /home/ywhuang/DML/IKS/experiment/dml3002/model/model_best_-1.pt
python dml_model_trans.py --cpu --save dmltest --sparseMode pw --t0 -12 --targetIKSSparsity 60 --decayDistance 10 --k 0.005 --model rcan --data_test Set5 --epochs 500 --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 3 --patch_size 144 --pre_train /home/ywhuang/DML/IKS/experiment/dml3003/model/model_best_-1.pt

python dml_model_trans.py --cpu --save dmltest --sparseMode pw --model rcan --reset --chop --n_resgroups 5 --n_resblocks 10 --n_feats 64 --scale 4 --patch_size 192 --pre_train /home/ywhuang/DML/IKS/experiment/dml6004/model/model_best_-1.pt
python dml_model_trans.py --cpu --save dmltest --sparseMode pw --model rcan --reset --chop --n_resgroups 5 --n_resblocks 10 --n_feats 64 --scale 3 --patch_size 144 --pre_train /home/ywhuang/DML/IKS/experiment/dml6003/model/model_best_-1.pt
python dml_model_trans.py --cpu --save dmltest --sparseMode pw --model rcan --reset --chop --n_resgroups 5 --n_resblocks 10 --n_feats 64 --scale 2 --patch_size 96  --pre_train /home/ywhuang/DML/IKS/experiment/dml6002/model/model_best_-1.pt

python dml_model_trans.py --cpu --save dmltest --sparseMode pw --template EDSR_paper --pre_train /home/ywhuang/DML/IKS/experiment/dml4007/model/model_best.pt --scale 2 --patch_size 96
python dml_model_trans.py --cpu --save dmltest --sparseMode pw --template EDSR_paper --pre_train /home/ywhuang/DML/IKS/experiment/dml4006/model/model_best.pt --scale 3 --patch_size 144
python dml_model_trans.py --cpu --save dmltest --sparseMode pw --template EDSR_paper --pre_train /home/ywhuang/DML/IKS/experiment/dml4005/model/model_best.pt --scale 4 --patch_size 192

python dml_model_trans.py --cpu --save dmltest --model NLSN --sparseMode pw --reset --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 4 --patch_size 192 --pre_train /home/ywhuang/DML/IKS/experiment/nlsn1002/model/model_best_-1.pt
python dml_model_trans.py --cpu --save dmltest --model NLSN --sparseMode pw --reset --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 3 --patch_size 144 --pre_train /home/ywhuang/DML/IKS/experiment/nlsn1003_f1/model/model_best_-1.pt
python dml_model_trans.py --cpu --save dmltest --model NLSN --sparseMode pw --reset --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 2 --patch_size 96  --pre_train /home/ywhuang/DML/IKS/experiment/nlsn1004_f2/model/model_best_-1.pt

python dml_model_trans.py --cpu --save dmltest  --init_decay 1.1 --lr 1e-5 --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 100  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 4 --patch_size 192 --pre_train ../../experiment/han1001_f2/model/model_best_-1.pt
python dml_model_trans.py --cpu --save dmltest  --init_decay 1.1 --lr 1e-5 --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 100  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 3 --patch_size 144 --pre_train ../../experiment/han1002_f2/model/model_best_-1.pt
python dml_model_trans.py --cpu --save dmltest  --init_decay 1.1 --lr 1e-5 --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 100  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 2 --patch_size 96  --pre_train ../../experiment/han1003_f2/model/model_best_-1.pt


# HAN
python main.py --whichgpu 1 --save han1001 --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 500  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 4 --patch_size 192 --pre_train ../../experiment/HAN_pretrain/HAN_BIX4.pt
python main.py --whichgpu 4 --save han1002 --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 500  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 3 --patch_size 144 --pre_train ../../experiment/HAN_pretrain/HAN_BIX3.pt --data_test Set14
python main.py --whichgpu 6 --save han1003 --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 500  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 2 --patch_size 96 --pre_train ../../experiment/HAN_pretrain/HAN_BIX2.pt --data_test Set14

# NLSN
python main.py --whichgpu 2 --save nlsn1002 --model NLSN --sparseMode pw --t0 -12 --targetIKSSparsity 80 --decayDistance 10 --k 0.005  --epochs 500  --lr_decay 200  --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 4 --patch_size 192 --pre_train ../../experiment/NLSN_pretrain/model_x4.pt
python main.py --whichgpu 2 --save nlsn1003 --model NLSN --sparseMode pw --t0 -12 --targetIKSSparsity 80 --decayDistance 10 --k 0.005  --epochs 500  --lr_decay 200  --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 3 --patch_size 144 --pre_train ../../experiment/NLSN_pretrain/NLSN_BIX3.pt  --data_test Set14
python main.py --whichgpu 2 --save nlsn1004 --model NLSN --sparseMode pw --t0 -12 --targetIKSSparsity 80 --decayDistance 10 --k 0.005  --epochs 500  --lr_decay 200  --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 2 --patch_size 96 --pre_train ../../experiment/NLSN_pretrain/NLSN_BIX2.pt  --data_test Set14

# finetune f1
python main.py --whichgpu 1 --save nlsn1004_f1 --model NLSN --sparseMode pw --t0 -12 --targetIKSSparsity 80 --decayDistance 10 --k 0.005  --epochs 100  --lr_decay 200  --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 2 --patch_size 96 --pre_train ../../experiment/nlsn1004/model/model_best_-1.pt  --data_test Urban100

python main.py --whichgpu 5 --save han1003_f1 --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 100  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 2 --patch_size 96 --pre_train ../../experiment/han1003/model/model_best_-1.pt --data_test Urban100
python main.py --whichgpu 1 --save han1002_f1 --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 100  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 3 --patch_size 144 --pre_train ../../experiment/han1002/model/model_best_-1.pt --data_test Urban100

python main.py --whichgpu 0 --save nlsn1003_f1 --model NLSN --sparseMode pw --t0 -12 --targetIKSSparsity 80 --decayDistance 10 --k 0.005  --epochs 100  --lr_decay 200  --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 3 --patch_size 144 --pre_train ../../experiment/nlsn1003/model/model_best_-1.pt  --data_test Urban100
python main.py --whichgpu 1 --save han1001_f1 --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 100  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 4 --patch_size 192 --pre_train ../../experiment/han1001/model/model_best_-1.pt --data_test Urban100

python main.py --whichgpu 1 --save nlsn1004_f2 --init_decay 1.1 --lr 1e-5 --model NLSN --sparseMode pw --t0 -12 --targetIKSSparsity 80 --decayDistance 10 --k 0.005  --epochs 100  --lr_decay 200  --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 2 --patch_size 96 --pre_train ../../experiment/nlsn1004/model/model_best_-1.pt  --data_test Urban100

# finetune f2
python main.py --whichgpu 1 --save han1001_f2  --init_decay 1.1 --lr 1e-5 --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 100  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 4 --patch_size 192 --pre_train ../../experiment/han1001/model/model_best_-1.pt --data_test Urban100+Set14
python main.py --whichgpu 0 --save han1002_f2  --init_decay 1.1 --lr 1e-5 --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 100  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 3 --patch_size 144 --pre_train ../../experiment/han1002/model/model_best_-1.pt --data_test Urban100+Set14
python main.py --whichgpu 1 --save han1003_f2  --init_decay 1.1 --lr 1e-5 --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 100  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 2 --patch_size 96 --pre_train ../../experiment/han1003/model/model_best_-1.pt --data_test Urban100+Set14

# finetune f3
python main.py --whichgpu 1 --save han1003_f3  --init_decay 1.1 --lr 1e-5 --model HAN --sparseMode pw --t0 -12 --targetIKSSparsity 65 --decayDistance 10 --k 0.005 --epochs 100  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 2 --patch_size 96 --pre_train ../../experiment/han1003/model/model_best_-1.pt --data_test Urban100+Set14

# gen visual result
# han x4
python main.py --whichgpu 0 --save hanx4_visual --model HAN --sparseMode han --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 500  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 4 --patch_size 192 --pre_train /home/ywhuang/DML/IKS/experiment/pretrain/HAN_pretrain/HAN_BIX4.pt --test_only --data_test Set5+Set14+B100+Urban100 --save_result
python main.py --whichgpu 0 --save hanx4_iks_visual --model HAN --sparseMode han --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005 --epochs 500  --lr_decay 200 --reset --chop --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 4 --patch_size 192 --pre_train /home/ywhuang/DML/IKS/Train_code/experiment/trans_model/hanx4_iks_trans.pt --test_only --data_test Set5+Set14+B100+Urban100 --save_result

# nlsn x4
python main.py --whichgpu 1 --save nlsnx4_visual --model NLSN --sparseMode nlsn --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005  --epochs 200  --lr_decay 200  --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 4 --patch_size 192 --pre_train /home/ywhuang/DML/IKS/experiment/pretrain/NLSN_pretrain/NLSN_BIX4.pt --test_only --data_test Set5+Set14+B100+Urban100 --save_result
python main.py --whichgpu 1 --save nlsnx4_iks_visual --model NLSN --sparseMode nlsn --t0 -12 --targetIKSSparsity 70 --decayDistance 10 --k 0.005  --epochs 200  --lr_decay 200  --reset --chop --rgb_range 1 --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 4 --patch_size 192 --pre_train /home/ywhuang/DML/IKS/Train_code/experiment/trans_model/nlsnx4_iks_trans.pt --test_only --data_test Set5+Set14+B100+Urban100 --save_result

