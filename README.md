# parkingDetection
This project is to detect available parking place with AI solution using sound data 
## Instruction
1. main.py file is the file to train, evaluate the model. Label "evaluate()" and dislabel "train(True/False)" to train the model. Use "train(False)" to rewrite the original model and "train(True)" to continue the previous training. The model will be saved in model.pkl under the same directory.
2. getDataParallel.py is the file to preprocess the data. Set "soundFileListDir" to be the directory containing all the sound data in wav format. Sound data in the desired directory will be preprocessed and the preprocessed result will be stored in "train_data.pkl" and "test_data.pkl" under the same directory if this two files do not exit. Our sound data, pretrained data file  and the detailed description can be download from here: <https://drive.google.com/drive/folders/1hyNJZvaR-QCbyKaIMFKXQ-TYELy7e-TK?usp=sharing>. 
3. To recover our work, copy model.pkl, train.pkl and test.pkl from "pretrain" directory to the same directory main.py lies in, then use "train(True)" to continue the training and "evaluate" to see the evaluation result with test data. 
