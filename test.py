import torch


# fixme: need to fix for testing
def test(model, test_dataloader, device=torch.device("cpu")):

    # model_checkpoint = torch.load(f'{args.proj}_checkpoint/{args.expname}/' + 'model_best.pth.tar')
    # model.load_state_dict(model_checkpoint['model_state_dict'])

    model.eval()
    num_correct = 0
    total = 0 # Total number of validation data
    for it in test_dataloader:
        x_batch, y_batch = it
        model = model.to(device)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(x_batch)

        pred_max = pred.max(dim=1)[1]

        n_correct = pred_max.eq(y_batch).sum().item()
        num_correct += n_correct
        total += len(pred_max)

    print(f"Accuracy: {num_correct/total}")