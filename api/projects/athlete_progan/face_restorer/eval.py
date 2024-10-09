from face_enhancement import FaceEnhancement
import cv2

processer = FaceEnhancement(in_size=256, model='GPEN-BFR-256', use_sr=True, device='cpu')

img = cv2.imread('temp.png', cv2.IMREAD_COLOR)
img_out, orig_faces, enhanced_faces = processer.process(img, aligned=False)
cv2.imwrite('temp_new.png', img_out)
cv2.imwrite('temp_new_2.png', enhanced_faces[0])
