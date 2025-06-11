import h5py 
import tqdm 
import random 

# use this script to combine hdf5 files and also to prune the data for experiment purposes 

sets_to_combine = [
                   "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_ood_goals/OOD_Goal_Door_Right.hdf5"
                   ]

output_file = "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_ood_goals/FILTERED_OOD_Goal_Door_Right.hdf5"
# rejects = [2, 5, 13, 14, 16, 19, 20, 35] # for the goal switch off 
# rejects = [4, 5, 18, 32, 33, 35, 37] # for the goal switch on 
# rejects = [3, 7, 18, 20, 25, 33, 36] # for door left 
# rejects = [0, 1, 3, 7, 17, 18, 20, 25, 27, 33, 37, 37] # for stricter door left 
# rejects = [0, 15, 22, 25, 29, 30, 38] # for door right
rejects = [6, 29, 30, 38] # for door right
# rejects = [0, 6, 15, 21, 22, 25, 29, 30, 38] # for stricter door right 
# rejects = [5, 12, 15, 20, 24, 31, 32, 36] # for button off 
# rejects = [2, 11, 12, 13, 14, 17, 19, 26, 27, 36] 

data_writer = h5py.File(output_file, 'w')
data_grp = data_writer.create_group("data")

cum_count = 0 
for datafile in sets_to_combine:
    current_count = 0 
    dataset = h5py.File(datafile, 'r')
    data_names = list(dataset["data"].keys())
    for demo in tqdm.tqdm(data_names):
        demo_number = int(demo.split("_")[1])
        if demo_number not in rejects:
            demo_grp = dataset["data"][demo]
            demo_grp.copy(demo_grp, data_grp, "demo_{}".format(cum_count))
            cum_count += 1 
            current_count += 1 
    print(current_count)
    dataset.close() 

data_writer.close() 