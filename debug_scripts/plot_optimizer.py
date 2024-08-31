from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
import matplotlib.pyplot as plt

model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-5)
num_sched_epochs = 40
warmup_epochs = 2
num_train_epochs = 25

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) 
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_sched_epochs)
scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=num_sched_epochs)

lrs = []

for i in range(num_train_epochs):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.ylabel("Learning rate")
plt.xlabel("Epoch")
plt.plot(lrs)
#plt.savefig("cosine-1cycle-epoch.pdf")
plt.show()