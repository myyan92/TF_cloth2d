import gin, os
from PIL import Image
from visual_inference.model_inference import ModelInference
from visual_inference.hybrid_inference import HybridInference
from visual_inference.fitting_inference import FittingInference
from TF_cloth2d.models.model_VGG_STN_2 import Model_STNv2
import numpy as np
import pdb

inference_type = 'hybrid'

if inference_type == 'model':
    gin.parse_config_files_and_bindings(['/scr-ssd/mengyuan/visual_inference/model_inference.gin'], [], finalize_config=False)
    gin.bind_parameter('ModelInference.snapshot', './train_real_ours_with_occlusion_pretrain_simseq_STNv2_consistency/model-45')
    inferencer = ModelInference()
elif inference_type == 'hybrid':
    gin.parse_config_files_and_bindings(['/scr-ssd/mengyuan/visual_inference/hybrid_inference.gin'], [], finalize_config=False)
    gin.bind_parameter('HybridInference.snapshot', './train_real_ours_with_occlusion_pretrain_simseq_STNv2_consistency/model-45')
    inferencer = HybridInference()
    inferencer.memory=False
else:
    inferencer = FittingInference()

for r in range(1,10):
  root = '/scr1/mengyuan/data/real_rope_ours_2/seq_m%d_2'%(r)
  #root = '/scr-ssd/mengyuan/TF_cloth2d/STN_imageloss/MPC_m_trial_0/iter_0_trial_1'
  actions = np.load(os.path.join(root, 'actions.npy'))
  actions = actions[:,[12,13,16,17]]
  i = 0
  prev_pred = None
  perceptions = []
  while os.path.exists(os.path.join(root, 'image_%d.png'%(i))):
    image = Image.open(os.path.join(root, 'image_%d.png'%(i)))
    image=np.array(image)
    if inference_type == 'fitting' and i==0:
        inferencer.set_guess(image)
    if i==0:
      for _ in range(4):
        pred_physical_state=inferencer.inference(image, render=False, prev_pred=prev_pred)
        prev_pred = pred_physical_state.copy()
        prev_pred[:,0] -= 0.5
        prev_pred[:,1] *= -1.0
        prev_pred = -prev_pred * 2.0

    pred_physical_state=inferencer.inference(image, render=False, prev_pred=prev_pred)
    prev_pred = pred_physical_state.copy()
    prev_pred[:,0] -= 0.5
    prev_pred[:,1] *= -1.0
    prev_pred = -prev_pred * 2.0
    perceptions.append(pred_physical_state)
    i += 1
    print(i)
  np.savez(os.path.join(root, 'history_perception.npz'), perception=np.array(perceptions), actions=actions)
