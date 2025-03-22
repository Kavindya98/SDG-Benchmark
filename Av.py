from openpyxl import load_workbook

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


def average_values(file_paths):
    num_files = len(file_paths)
    total_values = [0.0] * len(read_best_model_values(file_paths[0]))

    for file_path in file_paths:
        values = read_best_model_values(file_path)

        if values is not None:
            total_values = [total + value for total, value in zip(total_values, values)]

    average_values = [total / num_files for total in total_values]

    return average_values

def compute_variance(numbers):
    # Step 1: Calculate the Mean
    mean = sum(numbers) / len(numbers)
    
    # Step 2: Compute the Squared Differences
    squared_diffs = [(x - mean) ** 2 for x in numbers]
    
    # Step 3: Calculate the Variance
    variance = sum(squared_diffs) / len(numbers)
    
    return round(mean,5), round(variance,5)

def varinace(file_paths):

    ood_avg = []
    for file_path in file_paths:
        values = read_best_model_values(file_path)
        ood_avg.append((values[4]+values[6]+values[8])/3)

    ood_avg = [l*100 for l in ood_avg]

    return compute_variance(ood_avg)

def read_file_and_parse(filename):
    data = []

    with open(filename, 'r') as file:
        for line in file:
            line_data = line.strip().split(',')
            data.append(line_data)

    return data

def save_to_excel(filename, data):
    wb = load_workbook(filename)
    ws = wb.active
    ws.append(data)
    wb.save(filename)

filename = '/home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/Results/PACS/SagNet/ResNet18_consistant_with_paper/data.txt'
result = read_file_and_parse(filename)

for row in result:
    #file_paths = ["/home/kavindya/data/Model/Duplicate/TFS-ViT_Token-level_Feature_Stylization/Results/T2T14_false/sweep_drate_0.1_nlay_1/t123_s0/out.txt", "/home/kavindya/data/Model/Duplicate/TFS-ViT_Token-level_Feature_Stylization/Results/T2T14_false/sweep_drate_0.1_nlay_1/t123_s1/out.txt", "/home/kavindya/data/Model/Duplicate/TFS-ViT_Token-level_Feature_Stylization/Results/T2T14_false/sweep_drate_0.1_nlay_1/t123_s2/out.txt"]
    result_old = average_values(row)
    result_100 = [i * 100 for i in result_old]
    result = [ round(elem, 2) for elem in result_100 ]
    OOD = round((result[4]+result[6]+result[8])/3,2)
    OOD_avg, OOD_variance = varinace(row)
    Average = [result[2], OOD, OOD_avg, OOD_variance, round(result[2]-OOD,2), result[4], result[6], result[8]]
    print("Average values:", Average)

    # filename = ""
    # save_to_excel(filename, Average)