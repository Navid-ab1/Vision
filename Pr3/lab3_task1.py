import cv2
import numpy as np

buf = []
cap = cv2.VideoCapture('eggs.avi')
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('eggs-reverse.avi', fourcc, 30.0, (w, h))

while True:
    ret, I = cap.read()
    if not ret:
        break
    buf.append(I)
buf.reverse()
for i in buf:
    out.write(i)
out.release()
cap.release()

