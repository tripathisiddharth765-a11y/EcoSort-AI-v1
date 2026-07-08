import torch
import torch.nn as nn
import torch.optim as optim
from model import ResNet18
from dataset import train_loader,test_loader
model=ResNet18()
loss_function=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
best_accuracy=0
print("Traning Loop")
for epoch in range(20):
    running_loss=0.0
    correct_pred=0
    total=0
    accuracy=0
    model.train()
    for images,labels in train_loader:
      output=model(images)
      loss=loss_function(output,labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
      predictions=torch.argmax(output,dim=1)
      correct_pred+=(predictions==labels).sum().item()
      total+=labels.size(0)
    accuracy=(correct_pred/total)*100
    average_loss=running_loss/len(train_loader)
    print(f"Epoch : {epoch+1}/20")
    print(f"Avg loss : {average_loss}")
    print(f"Accuracy : {accuracy}")
     
torch.save(model.state_dict(),"waste_classifier.pth")












































