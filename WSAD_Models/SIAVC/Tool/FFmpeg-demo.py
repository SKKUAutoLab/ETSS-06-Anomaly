import os
import subprocess

input_directory = r'E:\迅雷下载\安全事故数据集（7类）\训练集\漏雨'
output_directory = r'E:\迅雷下载\安全事故数据集（7类）\训练集\漏雨\output'
os.makedirs(output_directory, exist_ok=True)
for filename in os.listdir(input_directory):
    if filename.endswith('.mp4'):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        command = f'ffmpeg -i "{input_path}" -r 15 "{output_path}"'
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"已处理文件: {filename}")
        except subprocess.CalledProcessError as e:
            print(f"处理文件 {filename} 时出错:", e)
print("所有视频处理完成。")