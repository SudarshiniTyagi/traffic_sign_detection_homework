# traffic_sign_detection_homework
The final results file(best result) is named `gtsrb_kaggle.csv`
The second best results file is named `gtsrb_kaggle_4_models_best.csv`

To generate final results file again on new data, run the following command:
```
python evaluate.py --model1 resnet_model_4/model_39.pth --model2 googlenet/model_18.pth --model3 resnet_34/model_13.pth --model4 stn_final_final/model_38.pth model5 resnet34_focal_loss/model_28.pth
```

# Abstract
This is a class project that describes all the experiments that have been carried out to classify images in The German Traffic Sign dataset. The goal of this project is to classify the signs into their respective categories with the highest accuracy possible. After all the experiments I was able to achieve a final and highest accuracy of 99.572. The report also discusses all the different models tried and their performance.

Full project report can be found [here](https://drive.google.com/file/d/1ZF7NpSZ2vZQ8zT4_1_9CHkBanoDFltpI/view?usp=sharing)

# Experiments and Results

\begin{small}
\begin{tabular}{L{4cm}C{1.5cm}R{1cm}}
Models & Type of Ensemble & Test Accuracy \\
\hline
\hline
ResNet-18, STN & Voting & 98.4 \\
\hline
ResNet-18, ResNet-18 with Focal loss & Average & 98.9\\ 
\hline
ResNet-18, ResNet-18(different checkpoint), Resnet-34 & Average & 99.065\\
\hline
ResNet-18, GoogLeNet, ResNet-34 & Average & 99.144\\
\hline
ResNet-18, GoogLeNet, ResNet-34, STN & Average & 99.477\\
\hline
ResNet-18, GoogLeNet, ResNet-34, STN, ResNet-34 with Focal loss & Average & 99.572\\

\end{tabular}
\end{small}


