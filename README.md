import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import copy

# Function to load and preprocess images
def image_loader(image_path, imsize=(512, 512)):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(torch.float).to(device)

# Display image
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title: plt.title(title)
    plt.axis('off')
    plt.show()

# Content & Style Loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images
content_img = image_loader("content.jpg")  # Replace with your content image
style_img = image_loader("style.jpg")      # Replace with your style image

assert content_img.size() == style_img.size(), \
    "Style and content images must be the same size"

# Load pretrained VGG19
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Normalization
normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean) / self.std

# Layers to compute losses
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Build the model
def get_style_model_and_losses(cnn, norm_mean, norm_std,
                                style_img, content_img):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(norm_mean, norm_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    return model, style_losses, content_losses

# Input image (copy of content image)
input_img = content_img.clone()
input_img.requires_grad_(True)

# Optimize
def run_style_transfer(cnn, norm_mean, norm_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, norm_mean, norm_std, style_img, content_img
    )
    optimizer = optim.LBFGS([input_img])

    print("Optimizing...")
    run = [0]
    while run[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}, Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")
            return loss
        optimizer.step(closure)

    return input_img.detach()

# Run transfer
output = run_style_transfer(cnn, normalization_mean, normalization_std,
                            content_img, style_img, input_img)

# Show output
imshow(output, title='Styled Output')
# Ai-project3
