# traffic_sign_detection_homework
The final results file(best result) is named `gtsrb_kaggle.csv`
The second best results file is named `gtsrb_kaggle_4_models_best.csv`

To generate final results file again on new data, run the following command:
```
python evaluate.py --model1 resnet_model_4/model_39.pth --model2 googlenet/model_18.pth --model3 resnet_34/model_13.pth --model4 stn_final_final/model_38.pth model5 resnet34_focal_loss/model_28.pth
```

