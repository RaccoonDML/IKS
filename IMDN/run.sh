python train_IMDN.py --scale 2 --pretrained checkpoints/IMDN_x2.pth
python train_IMDN.py --scale 3 --pretrained checkpoints/IMDN_x3.pth
python train_IMDN.py --scale 4 --pretrained checkpoints/IMDN_x4.pth --save imdn1001 --whichgpu 0

python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdn1001 --whichgpu 1 --nEpochs 200 --tar 50
python train_IMDN.py --scale 3 --patch_size 144 --pretrained checkpoints/IMDN_x3.pth --save imdn1002 --whichgpu 5 --nEpochs 200 --tar 50
python train_IMDN.py --scale 2 --patch_size 96  --pretrained checkpoints/IMDN_x2.pth --save imdn1003 --whichgpu 5 --nEpochs 200 --tar 50

python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_1 --whichgpu 5 --nEpochs 200 --tar 70
python train_IMDN.py --scale 3 --patch_size 144 --pretrained checkpoints/IMDN_x3.pth --save imdnx3_1 --whichgpu 5 --nEpochs 200 --tar 70
python train_IMDN.py --scale 2 --patch_size 96  --pretrained checkpoints/IMDN_x2.pth --save imdnx2_1 --whichgpu 5 --nEpochs 200 --tar 70

python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_t1 --whichgpu 1 --nEpochs 200 --tar 90
python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_t2 --whichgpu 3 --nEpochs 200 --tar 50
python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_t3 --whichgpu 5 --nEpochs 200 --tar 90

python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_2 --whichgpu 7 --nEpochs 200 --tar 30
python train_IMDN.py --scale 3 --patch_size 144 --pretrained checkpoints/IMDN_x3.pth --save imdnx3_2 --whichgpu 7 --nEpochs 200 --tar 30
python train_IMDN.py --scale 2 --patch_size 96  --pretrained checkpoints/IMDN_x2.pth --save imdnx2_2 --whichgpu 7 --nEpochs 200 --tar 30

python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_tar10 --whichgpu 5 --nEpochs 200 --tar 10
python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_tar20 --whichgpu 5 --nEpochs 200 --tar 20
python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_tar30 --whichgpu 5 --nEpochs 200 --tar 30
python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_tar40 --whichgpu 6 --nEpochs 200 --tar 40
python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_tar50 --whichgpu 6 --nEpochs 200 --tar 50
python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_tar60 --whichgpu 6 --nEpochs 200 --tar 60
python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_tar70 --whichgpu 7 --nEpochs 200 --tar 70
python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_tar80 --whichgpu 7 --nEpochs 200 --tar 80
python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_tar90 --whichgpu 7 --nEpochs 200 --tar 90
python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_tar99 --whichgpu 3 --nEpochs 200 --tar 99

python train_IMDN.py --scale 4 --patch_size 192 --pretrained checkpoints/IMDN_x4.pth --save imdnx4_3 --whichgpu 1 --nEpochs 250 --tar 30
python train_IMDN.py --scale 3 --patch_size 144 --pretrained checkpoints/IMDN_x3.pth --save imdnx3_3 --whichgpu 1 --nEpochs 250 --tar 30
python train_IMDN.py --scale 2 --patch_size 96  --pretrained checkpoints/IMDN_x2.pth --save imdnx2_3 --whichgpu 1 --nEpochs 250 --tar 30


# final model test
python test_IMDN.py --upscale_factor 4  --dataset Set5 --output_folder result/IMDN_IKS_x4 --checkpoint /home/mldai/workspace/IKS/IMDN/experiment/imdn1001/my_model_best.pth

python test_IMDN.py --upscale_factor 4  --dataset Set5 --output_folder result/IMDN_IKS_x4/Set5 --checkpoint /home/mldai/workspace/IKS/IMDN/checkpoints/IMDN_x4.pth --test_hr_folder /home/mldai/workspace/IKS/SRDATA/benchmark/Set5/HR --test_lr_folder /home/mldai/workspace/IKS/SRDATA/benchmark/Set5/LR_bicubic/X4

python test_IMDN_raw.py --test_hr_folder Test_Datasets/Set5/ --test_lr_folder Test_Datasets/Set5_LR/x2/ --output_folder results/Set5/x2 --checkpoint checkpoints/IMDN_x2.pth --upscale_factor 2

python test_IMDN.py --test_hr_folder Test_Datasets/Set5/ --test_lr_folder Test_Datasets/Set5_LR/x4/ --output_folder results/IMDN/Set5/x4 --checkpoint checkpoints/IMDN_x4.pth --upscale_factor 4
python test_IMDN.py --test_hr_folder Test_Datasets/Set5/ --test_lr_folder Test_Datasets/Set5_LR/x3/ --output_folder results/IMDN/Set5/x3 --checkpoint checkpoints/IMDN_x3.pth --upscale_factor 3
python test_IMDN.py --test_hr_folder Test_Datasets/Set5/ --test_lr_folder Test_Datasets/Set5_LR/x2/ --output_folder results/IMDN/Set5/x2 --checkpoint checkpoints/IMDN_x2.pth --upscale_factor 2

python test_IMDN.py --test_hr_folder Test_Datasets/Set5/ --test_lr_folder Test_Datasets/Set5_LR/x4/ --output_folder results/IMDN_IKS/Set5/x4 --checkpoint /home/mldai/workspace/IKS/IMDN/experiment/imdn1001/my_model_best.pth --upscale_factor 4
python test_IMDN.py --test_hr_folder Test_Datasets/Set5/ --test_lr_folder Test_Datasets/Set5_LR/x3/ --output_folder results/IMDN_IKS/Set5/x3 --checkpoint /home/mldai/workspace/IKS/IMDN/experiment/imdn1002/my_model_best.pth --upscale_factor 3
python test_IMDN.py --test_hr_folder Test_Datasets/Set5/ --test_lr_folder Test_Datasets/Set5_LR/x2/ --output_folder results/IMDN_IKS/Set5/x2 --checkpoint /home/mldai/workspace/IKS/IMDN/experiment/imdn1003/my_model_best.pth --upscale_factor 2

python test_IMDN.py --save IMDN --checkpoint checkpoints/IMDN_x4.pth --upscale_factor 4 --dataset Set14
python test_IMDN.py --save IMDN --checkpoint checkpoints/IMDN_x3.pth --upscale_factor 3 --dataset Set14
python test_IMDN.py --save IMDN --checkpoint checkpoints/IMDN_x2.pth --upscale_factor 2 --dataset Set14

python test_IMDN.py --save IMDN_IKS40 --checkpoint experiment/imdn1001/my_model_best.pth --upscale_factor 4 --dataset Set5
