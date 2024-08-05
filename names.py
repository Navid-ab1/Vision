import os
def file_name(folder_path):
    file_name=[]
    for i in os.listdir(folder_path):
        file_name.append(i)
    return file_name
folder_path1= '/home/navid/Desktop/VisionProject/Dataset'
folder_path2 = '/home/navid/Desktop/VisionProject/Yolo'
print(file_name(folder_path1),file_name(folder_path2))