from visual_inference.fitting_inference import FittingInference
from PIL import Image
import numpy as np

inferencer=FittingInference()
#im=Image.open('robustness_image_1.jpg')
#im=Image.open('robustness_image_4.jpg')
#im=Image.open('./playground/sample_distractor_image_5.png')
#im=Image.open('/scr1/mengyuan/data/real_rope_with_occlusion-new/run_1/08.png')
#im=Image.open('./gen_data/real_rope/run04/img_0000.jpg')
im=Image.open('./gen_data/real_rope/run60/img_0093.jpg')
#im=Image.open('./playground/image_loss/real_syc_2.png')
im=im.resize((200,200), Image.LANCZOS)
im=np.array(im)
inferencer.set_guess(im)
inferencer.inference(im)
start=Image.open('00.png')
start.show()
start=Image.open('2000.png')
start.show()

