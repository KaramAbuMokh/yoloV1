image_shape = (448, 448, 3)
grid_width = 7
grid_height = 7
cell_grid_width = image_shape[0] / grid_width
cell_grid_height = image_shape[1] / grid_height
boxes_in_cell = 2
data_dir = '/kitti_single/training'
labels = ['Car', 'Van', 'Truck']