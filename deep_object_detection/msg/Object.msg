string label   # label of the object
uint32 x       # x coordinate of the anchor position of bounding box (top-left)
uint32 y       # y coordinate of the anchor position of bounding box (top-left)
uint32 width   # width of the bounding box
uint32 height  # height of the bounding box
uint32 imageID # the id of the image that the object belongs to
geometry_msgs/Point[] mask  # the mask obtained from position sensitive score maps (Valid only for res-net) 
float32 confidence # the confidence value of the neural net
