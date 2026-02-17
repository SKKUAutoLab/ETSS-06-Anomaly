import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Extract a video to frames')
parser.add_argument('--video_dir', default='demo.mp4', type=str)
parser.add_argument('--folder_extract', default='demo', type=str)
args = parser.parse_args()

if not os.path.exists(args.folder_extract):
   os.mkdir(args.folder_extract)
cam = cv2.VideoCapture(args.video_dir)
cnt = 0
while True:
   ret, frame = cam.read()
   if ret:
      if cnt < 10:
         name = args.folder_extract + '/0000' + str(cnt) + '.jpg'
      elif cnt >= 10 and cnt < 100:
         name = args.folder_extract + '/000' + str(cnt) + '.jpg'
      elif cnt >= 100 and cnt < 1000:
         name = args.folder_extract + '/00' + str(cnt) + '.jpg'
      elif cnt >= 1000 and cnt < 10000:
         name = args.folder_extract + '/0' + str(cnt) + '.jpg'
      else:
         name = args.folder_extract + '/' + str(cnt) + '.jpg'
      print ('Created frame ' + name)
      img = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
      cv2.imwrite(name, img)
      cnt += 1
   else:
      break
cam.release()
cv2.destroyAllWindows()