
import argparse


def read_best_model_values(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    last_best_model_index = max([i for i, line in enumerate(lines) if "Best model upto now" in line], default=-1)

    if last_best_model_index == -1:
        print(f"No 'Best model up to now' found in {file_path}")
        return None
    values_line = lines[last_best_model_index + 1].strip()

    values_list = [float(value) for value in values_line.split()]
    return values_list


def average_over_trials(file_paths):
    num_files = len(file_paths)
    total_values = [0.0] * len(read_best_model_values(file_paths[0]))

    for file_path in file_paths:
        values = read_best_model_values(file_path)

        if values is not None:
            total_values = [total + value for total, value in zip(total_values, values)]

    average_values = [total / num_files for total in total_values]

    return average_values


def read_file_and_parse(filename, trial_seed=[0, 1, 2]):
    data = []

    for trail in trial_seed:
        file_path = filename + f"/t123_s{trail}/out.txt"
        data.append(file_path)

    return data

def main():
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--filename', type=str, default="./Results/PACS_Custom/ME_ADA_CNN/Resnet18")
    args = parser.parse_args()

    
    file_paths = read_file_and_parse(args.filename)

    
    result_old = average_over_trials(file_paths)
    result_100 = [i * 100 for i in result_old]
    result = [ round(elem, 2) for elem in result_100 ]
    OOD = round((result[4]+result[6]+result[8])/3,2)
    print("IID (Photo) Performance:", result[2])
    print("Overall OOD Performance:", OOD)
    print("Domain Gap:",round(result[2]-OOD,2))
    print("OOD (Art) Performance:", result[4])
    print("OOD (Cartoon) Performance:", result[6])
    print("OOD (Sketch) Performance:", result[8])


if __name__ == "__main__":
    main()