import torch
import torch.nn as nn
import torch.optim as optim

import helper_utils

torch.manual_seed(42)
# distances in miles from recent travels
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
# corresponding commutime times in minutes
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

# a single neuron with one input uses a linear equation: Time = W * Distance + B
model = nn.Sequential(nn.Linear(1, 1))
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    optimizer.zero_grad() 
    outputs = model(distances)
    loss = loss_function(outputs, times) # calculate the loss
    loss.backward() 
    optimizer.step() # update the model's parameters
    # print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

helper_utils.plot_results(model, distances, times) # plotting model predictions

distance_to_predict = 7.0 # what expected time for 7 miles

with torch.no_grad():
    new_distance = torch.tensor([[distance_to_predict]], dtype=torch.float32)
    predicted_time = model(new_distance) # pass new data to trained model to get a prediction

    # use .item() to extract the scalar value from the tensor for printing
    print(f"\nPrediction for a {distance_to_predict}-mile deliver: {predicted_time.item():.1f} minutes")

    if predicted_time.item() > 30:
        print("\nDecision: Do NOT take the job. You will probably be late.")
    else:
        print("\nDecision: Take the job. You can make it!")

### Evaluation
layer = model[0]
weights = layer.weight.data.numpy()
bias = layer.bias.data.numpy()

print(f"Weight: {weights}")
print(f"Bias: {bias}")
