from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
from usefultools import UsefulTools
    

classes = ['France', 'Spain', 'Ukraine']
types = ['train', 'test', 'valid']
def build_model(train_path, test_path, valid_path):

    
    train_batches = ImageDataGenerator(rescale = 1./ 255).flow_from_directory(directory=train_path, target_size = (224,224),
                                classes = classes, batch_size = 10)
    valid_batches = ImageDataGenerator(rescale = 1./ 255).flow_from_directory(directory=valid_path, target_size = (224,224),
                                classes = classes, batch_size = 4)
    test_batches = ImageDataGenerator(rescale = 1./ 255).flow_from_directory(directory=test_path, target_size = (224,224),
                                classes = classes, batch_size = 140)
    
    imgs, labels = next(train_batches)
        
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (224, 224, 3)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(3, activation = 'softmax'))
    return train_batches, valid_batches, test_batches, model


def train(train_batches, valid_batches, model, saving_path):
    
    print ("training...")
    model.compile(optimizer=Adam(learning_rate = 0.0001),
                  loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    model.fit_generator(generator = train_batches, steps_per_epoch = 63, 
                    validation_data = valid_batches, validation_steps = 65, epochs = 5, verbose = 1)
    
    
    model.save(saving_path)
    

def test(test_batches, model):
    helper = UsefulTools
    test_imgs, test_labels = next(test_batches)
    predictions = model.predict_generator(generator = test_batches, steps = 1, verbose = 0)
    accuracy_sum = 0
    for index, prediction in enumerate(predictions):
    
        ind = helper.get_max(prediction)
        country_index = helper.get_max(test_labels[index])
        acctual_country = classes[country_index]
        print ("Country: {0}".format(acctual_country), end = '')
        predicted_country = classes[ind]
        helper.make_spaces(acctual_country)
        print ("Prediction: {0}".format(predicted_country))
        if acctual_country == predicted_country:
            accuracy_sum += 100


    print ("\nAccuracy precentage: {0}%".format((accuracy_sum / (index + 1))))
    
def main():
    folder_paths = []
    hello_message = "\nHello and welcome to flag identification!\n\n"
    folder_error = "\nThe given path does not exists, please try again"
    error_message = "\nSorry, did not understand your request, please try again"
    load_check = "\nDo you have\want to enter a path that leads to a saved model? write 'yes' or 'no'"
    load_error = "\nThe given path does not contain the saved model, please try again"
    did_not_load_error = "\nBecause you did not load the model, you can not predict at the moment, sorry.."
    save_message = "\nPlease write the path you want to save the model in: "
    folder_warning = "\nThe path is valid, but the name of the folder is not in it, are you sure it is the right path? write 'yes' or 'no'"
    helper = UsefulTools
    i = 0
    
    helper.print_message(hello_message)
    
    while True:
      flag = False
      helper.print_message("\nPlease enter the path to the {0} folder: ".format(types[i]))
      folder_path = input()
      if helper.check_path(folder_path):
          if not helper.find_word(types[i], folder_path):
              while True:   
                  
                    helper.print_message(folder_warning)
                    choice = input()
                    if choice == 'no':
                        flag = True
                    break
          if flag:
            continue
    
          folder_paths.append(folder_path)
          i += 1
                      
      else:
          helper.print_message(folder_error)
          
      if i == 3:
          break
    
    flag = False
    train_batches, valid_batches, test_batches, model = build_model(folder_paths[0], folder_paths[1], folder_paths[2])
    
    helper.print_message(load_check)
    choice = input()
    if choice == 'yes':
        flag = True
        
        while True:
            
            helper.print_message("\nPlease write the path to the saved model: ")
            load_path = input()
        
            if helper.check_path(load_path):
                
                helper.print_message("loading the weigths...")
                model = load_model(load_path)
                break
                
            else:
                
                helper.print_message(load_error)
        
    while True:
        
        choice = input("Please write your choice. train or predict: ")
        
        if choice == 'train':
            while True:
                
                helper.print_message(save_message)
                saving_path = input()
                if helper.check_path(saving_path):
                    break
                else:
                    helper.print_message(folder_error)
            train(train_batches, valid_batches, model, saving_path)
            break
        
        if choice == 'predict':
            if flag == True:
                
                test(test_batches, model)
                break
            
            else:
                helper.print_message(did_not_load_error)
                continue
            
        helper.print_message(error_message)
    
if __name__ == '__main__':
    main()
