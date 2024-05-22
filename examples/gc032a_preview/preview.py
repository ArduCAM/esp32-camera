import numpy as np
import cv2

np.set_printoptions(formatter={'int':hex})

def YUVToMat(data,Width,Height, color_mode):
    codeMap = {
        0:cv2.COLOR_YUV2BGR_YUYV,
        1:cv2.COLOR_YUV2BGR_YVYU,
        2:cv2.COLOR_YUV2BGR_UYVY,
        3:cv2.COLOR_YUV2BGR_Y422,
        4:cv2.COLOR_YUV2BGR_YUY2,
    }
    image = np.frombuffer(data, np.uint8).reshape(Height, Width, 2)
    # print(image.shape)
    return cv2.cvtColor(image,codeMap[color_mode])

def RGB565ToMat(data,Width,Height):
    arr = np.frombuffer(data,dtype='<u2').astype(np.uint32)
    arr = ((arr & 0xFF00) >> 8) + ((arr & 0x00FF) << 8)
    arr = 0xFF000000 + ((arr & 0xF800) << 8) + ((arr & 0x07E0) << 5) + ((arr & 0x001F) << 3)

    arr.dtype = np.uint8
    image = arr.reshape(Height,Width,4)
    # return cv2.flip(image,0)
    return image

data = []

# with open("frame_1_0220.txt") as f:
#     for line in f:
#         l = line.strip().split(",")
#         data.append(int(l[3], 16))

with open("320x240_rgb.txt") as f:
    for line in f:
        l = line.strip().split(" ")
        for ch in l:
            if ( ch == ''):
                continue
            data.append(int(ch,16))
data = np.array(data)
# print("data: {}".format(data))

# header = data[:9]
# # print("header: {}".format(header))
# data = data[9:]

# lines = []

# while data.shape[0] > 240*2:
# while data.shape[0] > 240*2:
#     line_header = data[:12]
#     # print(line_header[4:6])
#     data = data[12:]
#     lines.append(data[:240*2])
#     # print(data[:240*2])

#     data = data[240*2:]

    # if len(lines) >= 320:
    #     break

# print(data.shape)
# print("end: {}".format(data))

img = np.array(data, dtype=np.uint8)
# print(img.shape)
# print(img)
# img.tofile("frame_2_0220_img.yuv")
# yuv1 = YUVToMat(img, 240, 320, 0)
# yuv2 = YUVToMat(img, 240, 320, 1)
# yuv3 = YUVToMat(img, 240, 320, 2)
# yuv4 = YUVToMat(img, 240, 320, 3)
# yuv5 = YUVToMat(img, 240, 320, 4)

rgb = RGB565ToMat(img,320,240)
# print("lines: {}".format(len(data)))

cv2.imshow("gc032a_rgb_preview", rgb)
# cv2.imshow("yuv1", yuv1)
# cv2.imshow("yuv2", yuv2)
# cv2.imshow("yuv3", yuv3)
# cv2.imshow("yuv4", yuv4)
# cv2.imshow("yuv5", yuv5)

cv2.waitKey()
# print(data[240*2:][:12])