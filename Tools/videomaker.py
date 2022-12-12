import os
import cv2

image_dir = "/Users/mac/PycharmProjects/videomaker/frames"
# print(os.path.exists(image_dir))
image_list = ([name for name in os.listdir(image_dir) if name.endswith('.jpg')])
image_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
path = "/Users/mac/PycharmProjects/videomaker/frames/frame1.jpg"
# print(os.path.exists(path))
img = cv2.imread(path)
dimensions = img.shape
fps = 60
size = (256, 256)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter("/Users/mac/PycharmProjects/videomaker/video/output.mp4", fourcc, fps, size, isColor=True)
for image_name in image_list:
    image_path = os.path.join("/Users/mac/PycharmProjects/videomaker/frames", image_name)
    frame = cv2.imread(image_path)
    video.write(frame)
video.release()
cv2.destroyAllWindows
