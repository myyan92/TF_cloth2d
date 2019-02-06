from sim_integration.hybrid_inference import HybridInference
from TF_cloth2d.models.model_VGG_STN_2 import Model_STNv2
from PIL import Image
import numpy as np
import glob

files = glob.glob('/scr1/mengyuan/data/real_rope_ours_2/seq_m3/image_[0-9]*.png')
model = Model_STNv2(vgg16_npy_path='/scr-ssd/mengyuan/TF_cloth2d/models/vgg16_weights.npz',
                    fc_sizes=[1024,1024,256],
                    save_dir='./tmp')
inferencer = HybridInference(model,
                             pred_target='node',
#                             snapshot='/scr-ssd/mengyuan/TF_cloth2d/simseq_data_pred_node_STN_2_large_augmented/model-38',
                             snapshot='/scr-ssd/mengyuan/TF_cloth2d/STN_imageloss/train_real_ours_pretrain_simseq_STNv2_augmented/model-47',
                             memory=False)

for f in files:
    im = Image.open(f)
    state  = inferencer.inference(np.array(im))
