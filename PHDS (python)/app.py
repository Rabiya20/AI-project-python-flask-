# -----------------------------------------LIBRARIRES------------------------------------------------
from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
from keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.image as mpimg

# ----------------------------------------------CNN LAYERS--------------------------------------------
model = Sequential()

model.add(Conv2D(16, (3,3), input_shape= (256,256,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# -------------------------------------LOAD SAVED MODEL----------------------------------------------
model.load_weights('static/PHDS_classification_model.h5')

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1


# --------------------------------------------MAIN WEBSITE PAGE-------------------------
@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT

    # # -----------------------------------IMAGE SAVE AND PREPROCESSING----------------------
    img = request.files['image']
    img.save('static/{}.jpg'.format(COUNT))
    test_image = image.load_img('static/{}.jpg'.format(COUNT), target_size = (256,256))    
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    pred = model.predict(test_image)
    COUNT+=1
    
    # # classes = {'healthy': 0, 'unhealthy': 1}
    # # ------------------------------------JUMP TO PREDICTION PAGE--------------------

    return render_template('prediction.html', data = pred )


# -----------------------------------EXTRAAAAAAAAAAAAAAAAAA

    # img = mpimg.imread(image_path)
    # plt.imshow(img)
    # if probabilities > 0.5:
    #     classes = "%.2f" % (probabilities[0]*100) + "% unhealthy"
    #     # plt.title("%.2f" % (probability[0]*100) + "% unhealthy")
    # else:
    #     classes = "%.2f" % (1-probabilities[0]*100) + "% healthy"
    #     # plt.title("%.2f" % ((1-probability[0])*100) + "% healthy")
    # # plt.show()
# --------------------EXTRA KHATAM
    

@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static/', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)