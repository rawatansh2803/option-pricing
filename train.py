# ==========================================
# FILE: train.py
# ==========================================
def the_trainer(model, num_epochs, train_loader, loss_fn, optimizer):
    model.train()
    for epoch in range(num_epochs):
        l_sum = 0
        for batch in train_loader:
            optimizer.zero_grad()
            X, y, meta = batch['X'].to(config.device), batch['y'].to(config.device), batch['meta'].to(config.device)
            loss = loss_fn(model(X, meta), y)
            loss.backward(); optimizer.step()
            l_sum += loss.item()
        if (epoch+1) % 10 == 0: print(f"Epoch {epoch+1} Loss: {l_sum/len(train_loader):.6
