import h5py 
import tqdm 
import random 

# use this script to combine hdf5 files and also to prune the data for experiment purposes 

sets_to_combine = [
                #     "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/button_off.hdf5",
                    # "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/button_on.hdf5",  
                #     "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/switch_off.hdf5",
                    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/switch_on.hdf5",
                    # "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/drawer_close.hdf5",
                   "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/drawer_open.hdf5",
                #    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/door_left.hdf5",
                #    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/door_right.hdf5",
                #    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/pink_lift.hdf5",
                #    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/red_lift.hdf5",
                # "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/blue_lift.hdf5"
                   ]

output_file = "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_combined_categories_wcubes/switch_on_drawer_open.hdf5"
# output_file = "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/blue_lift_80.hdf5"
accept_proportion = 1 # this allows you to randomly accept / reject demos 
accept_cap = 20 # this allows you to collect N demos from each 

data_writer = h5py.File(output_file, 'w')
data_grp = data_writer.create_group("data")

cum_count = 0 
for datafile in sets_to_combine:
    current_count = 0 
    dataset = h5py.File(datafile, 'r')
    data_names = list(dataset["data"].keys())
    random.shuffle(data_names)
    for demo in tqdm.tqdm(data_names):
        demo_grp = dataset["data"][demo]
        demo_grp.copy(demo_grp, data_grp, "demo_{}".format(cum_count))
        current_count += 1 
        cum_count += 1 
        if accept_cap is not None and current_count >= accept_cap:
            break 
    print(current_count)
    dataset.close() 

data_writer.close() 