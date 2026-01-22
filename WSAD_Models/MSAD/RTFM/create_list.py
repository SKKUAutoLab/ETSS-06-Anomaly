input_file_path = "list/shanghai-i3d-train-10crop.list"
output_file_path = "list/shanghai-swin-test-10crop-1.list"

with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
    for line in input_file:
        new_line = "/scratch/kf09/lz1278/SHT-Swin/" + line.strip().split("/")[-1].replace("i3d", "swin")
        print(new_line)