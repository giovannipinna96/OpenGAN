import torch

def evalutate_data(netD, dataloader, device):
    correct = 0
    netD.train()
    outputs = []
    with torch.no_grad():
        for data in dataloader:
            output = netD(data.to(device)).view(-1)
            outputs += [output.cpu()]
            correct += (output >= .5).sum().item()

    return torch.cat(outputs), correct

def test_model(backbone, testloader, netD, device):

    x = next(iter(testloader)).to(device)
    (netD(x).view(-1) >= .5).sum().item() / x.size(0)

    outputs_close, correct = evalutate_data(netD, testloader, device)
    outputs_close = outputs_close.numpy()

    print("Correctly identified items:", correct/len(testloader.dataset))

    correct_fake = 0
    outputs_open = []
    with torch.no_grad():
        for ite in range(10):
            features = []
            noiseimg = torch.randn(100, 3, 256, 256, device=device)
            
            def get_features(module, input_, output):
              features.append(output.cpu().detach())
            handle = backbone.layer4.register_forward_hook(get_features)
            
            _ = backbone(noiseimg)
            feats = features[0].to(device)
            assert feats.shape == (100, 512, 8, 8), f"Features shape is {feats.shape}, expected (100, 512, 8, 8)"
            output = netD(feats).view(-1)
            outputs_open.append(output.cpu())
            correct_fake += (output < .5).sum().item()
    outputs_open = torch.cat(outputs_open).numpy()

    print("Correctly identified fakes:", correct_fake/1000)

    return outputs_open, outputs_close
