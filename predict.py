# FILE: predict.py
# ==========================================
def the_predictor(model, list_of_paths):
    model.eval()
    results = []
    with torch.no_grad():
        for path in list_of_paths:
            data = np.load(path, allow_pickle=True).item()
            X = torch.tensor(data['X']).unsqueeze(0).to(config.device)
            meta = torch.tensor(data['meta']).unsqueeze(0).to(config.device)
            norm_tv = model(X, meta).cpu().numpy()[0]
            price = (norm_tv * data['spot']) + data['intrinsic']
            results.append(price)
    return results
