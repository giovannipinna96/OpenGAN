import torch

def test_model(backbone, testloader, netD, device):

    x = next(iter(testloader)).cuda()
    (netD(x).view(-1) >= .5).sum().item() / x.size(0)

    correct = 0
    netD.eval()

    outputs_close = []
    with torch.no_grad():
        for test_data in testloader:
            output = netD(test_data.cuda()).view(-1)
            outputs_close += [output.cpu()]
            correct += (output >= .5).sum().item()

    outputs_close = torch.cat(outputs_close).numpy()

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
            feats = features[0].cuda()
            # TODO is correct -> assert feats.shape == (100, 512, 8, 8), f"Features shape is {feats.shape}, expected (100, 512, 8, 8)"
            assert feats.shape == (100, 2048, 8, 8), f"Featu"
            output = netD(feats).view(-1)
            outputs_open.append(output.cpu())
            correct_fake += (output < .5).sum().item()
    outputs_open = torch.cat(outputs_open).numpy()

    print("Correctly identified fakes:", correct_fake/1000)

    return outputs_open, outputs_close
