from unet import *
from data import *

mydata = dataProcess(512,512)
mydata.test_path = "./data/c1"
mydata.create_test_data()       # create test data for c3
imgs_test = mydata.load_test_data()

myunet = myUnet()
model = myunet.get_unet()
model.load_weights('./results/unet.hdf5')
imgs_mask_test = model.predict(imgs_test, batch_size=5, verbose=1)
np.save('./results/imgs_mask_test.npy', imgs_mask_test)

for i in range(imgs_mask_test.shape[0]):
    img = imgs_mask_test[i]
    print('184:', img.max(), img.min())
    img = array_to_img(img)
    img.save("./results/%d.jpg" % (i))