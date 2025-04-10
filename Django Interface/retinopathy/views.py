from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
import os

import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

def load_model(model_path, num_classes=5):
    model = models.efficientnet_b3(pretrained=False)

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),  
        nn.Linear(model.classifier[1].in_features, num_classes)
    )

   
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

   
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == 'classifier.1.1.weight':
            new_state_dict['classifier.1.weight'] = value
        elif key == 'classifier.1.1.bias':
            new_state_dict['classifier.1.bias'] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) 
    return image

CLASS_LABELS = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferative_DR'
}

def classify_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        return predicted_class, CLASS_LABELS[predicted_class]  

state_dict = torch.load('retinopathy\\fineTunedEfficientnet_b3.pt', map_location=torch.device('cpu'))

# Print the keys
# for key in state_dict.keys():
#     print(key)
model = load_model('retinopathy\\fineTunedEfficientnet_b3.pt')

def home(request):
    if request.method == 'POST' and request.FILES['image']:
     
        image_file = request.FILES['image']
        file_path = default_storage.save(image_file.name, image_file)
        full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)

        image_tensor = preprocess_image(full_file_path)
        predicted_class, predicted_label = classify_image(model, image_tensor)

        default_storage.delete(file_path)

        return render(request, 'result.html', {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label
        })

    return render(request, 'home.html')