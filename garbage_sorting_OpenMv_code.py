# Edge Impulse - OpenMV Image Classification Example

import sensor, image, time, os, tf, uos, gc

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None

try:
    # Load built in model
    labels, net = tf.load_builtin_model('trained')
except Exception as e:
    raise Exception(e)


clock = time.clock()

def find_greatest_number(num1, num2, num3, num4):
    max_num=num1;

    if (num2> max_num):
        max_num=num2

    if (num3>max_num):
        max_num = num3

    if(num4>max_num):
        max_num=num4

    return max_num



while(True):
    clock.tick()

    img = sensor.snapshot()

    # default settings just do one detection... change them to search the image...
    for obj in net.classify(img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        img.draw_rectangle(obj.rect())
        # This combines the labels and confidence values into a list of tuples
        predictions_list = list(zip(labels, obj.output()))

#        for i in range(len(predictions_list)):
#            max_num=predictions_list[i][1];

#            print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))

    data=find_greatest_number(predictions_list[0][1],predictions_list[1][1],predictions_list[2][1],predictions_list[3][1])
    key=obj.output().index(data)

    print("%s = %f" % (predictions_list[key][0],data))






    print(clock.fps(), "fps")
