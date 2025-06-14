import torchvision
import dill
from copy import deepcopy
from tqdm import tqdm
from pytorch_grad_cam import (
    GradCAM, 
    HiResCAM, 
    ScoreCAM, 
    GradCAMPlusPlus, 
    AblationCAM, 
    XGradCAM, 
    FullGrad, 
    EigenGradCAM, 
    LayerCAM
)
from fastai.vision.all import *
from fastai.vision import *
def original_cam(model, input_tensor, label):
    cam = torch.einsum('ocij,kc -> okij', model[0](input_tensor), model[1][2].weight).data.cpu()
    cam = cam[0,0,:,:] if label == 0 else cam[0,1,:,:]
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0), 
        size=(512, 512), 
        mode='bilinear', 
        align_corners=False
    )
    return cam.squeeze(0)
def make_figure(img,allcams):
    fig, axs = plt.subplots(3,4)
    for ax in axs.flatten():
        img.show(ax=ax)
    axs[0][0].set_title("Original Image")
    axs[0][1].set_title("H-CAM (proposed)")
    axs[0][2].set_title("CAM")
    for ax,method in zip(axs.flatten()[3:], METHODS):
        ax.set_title(f"{method.__name__}")
    #
    for ax,cam in zip(axs.flatten()[1:], allcams):
        ax.imshow(cam.squeeze(), alpha=0.7, cmap="magma")
    fig.set_figwidth(10)            
    fig.set_figheight(7.5)
    fig.tight_layout() 
    return fig
def get_img_and_allcams(dls,idx,model):
    img, label = dls.train_ds[idx]
    img_norm, = next(iter(dls.test_dl([img])))
    cam = original_cam(model=model, input_tensor=img_norm, label=label)
    hcam = original_cam(model=model, input_tensor=img_norm, label=label)
    allcams = [cam, hcam]
    for method in METHODS:
        allcams.append(torch.tensor(method(model=model, target_layers=model[0][-1])(input_tensor=img_norm,targets=None)))
    return img, allcams
def get_img_and_originalcam(dls,idx,model):
    img, label = dls.train_ds[idx]
    img_norm, = next(iter(dls.test_dl([img])))
    cam = original_cam(model=model, input_tensor=img_norm, label=label)
    return img, cam
#---#
METHODS = (
    GradCAM, 
    HiResCAM, 
    ScoreCAM, 
    GradCAMPlusPlus, 
    AblationCAM, 
    XGradCAM, 
    FullGrad, 
    EigenGradCAM, 
    LayerCAM
)
THETA = 0.05 # 0.05로 하는게 나을듯
#---#
PATH = f'./data/chest_xray_THETA{THETA}/3hcam/originalsize'
dls = ImageDataLoaders.from_name_func(
    path = PATH,
    fnames = get_image_files(PATH),
    label_func = lambda x: "NORMAL" if x[0].isupper() else "PNEUMONIA",
    item_tfms = Resize(512),
    batch_tfms= ToTensor(),
    num_workers = 0
)
compose = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((512,512)),
        torchvision.transforms.ToPILImage()
])

for idx, _ in enumerate(dls.train_ds):
    img, label = dls.train_ds[idx]
    img_tensor = compose(img)    
    fname = str(dls.train_ds.items[idx]).split("/")[-1]
    img_tensor.save(f"./data/chest_xray_THETA{THETA}/3hcam/removed0/{fname}")
#---#
dls_list = []
lrnr_list = []
for i in range(3):
    PATH = f'./data/chest_xray_THETA{THETA}/3hcam/removed{i}'
    torch.manual_seed(43052)
    dls = ImageDataLoaders.from_name_func(
        path = PATH,
        fnames = get_image_files(PATH),
        label_func = lambda x: "NORMAL" if x[0].isupper() else "PNEUMONIA",
        item_tfms = Resize(512),
        batch_tfms= ToTensor(),
        num_workers = 0
    )
    dls_list.append(dls)
    lrnr = vision_learner(dls,resnet101,metrics=error_rate)
    lrnr_list.append(lrnr)
    lrnr.model[1] = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(output_size=1), 
        torch.nn.Flatten(),
        torch.nn.Linear(2048,out_features=2,bias=False)
    )
    lrnr.fine_tune(3)
    for idx, _ in enumerate(dls.train_ds):
        img, cam = get_img_and_originalcam(dls=dls,idx=idx,model=lrnr.model)
        compose = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((512,512))
        ])
        img_tensor = torchvision.transforms.ToTensor()(img)
        k = THETA/cam.std()**2 
        weight = np.exp(-k*cam)
        res_img_tensor= img_tensor*weight / (img_tensor*weight).max()
        res_img = torchvision.transforms.ToPILImage()(res_img_tensor)
        fname = str(dls.train_ds.items[idx]).split("/")[-1]
        res_img.save(f"./data/chest_xray_THETA{THETA}/3hcam/removed{i+1}/{fname}")
#---#
dls = dls_list[0]
camdata = []
for idx, path in enumerate(dls.train_ds.items):
    lrnr = lrnr_list[0]    
    img, allcams = get_img_and_allcams(dls=dls,idx=idx,model=lrnr.model)
    hcams = [allcams[0]]
    for lrnr in lrnr_list[1:]:
        _, cam = get_img_and_originalcam(dls=dls,idx=idx,model=lrnr.model)
        hcams.append(cam)
    allcams[0] = 0.434*hcams[0] +\
                0.3*hcams[1] +\
                0.266*hcams[2]
    fig = make_figure(img,allcams)
    fname = str(path).split("/")[-1].split(".")[0]
    camdata.append([fname,allcams,hcams]) #
    fig.savefig(f"./figs/chest_THETA{THETA}/{fname}-3hcam.pdf")
with open(f'results/chest/THETA{THETA}/dls_list-3hcam.pkl', 'wb') as f:
    dill.dump(dls_list, f)
with open(f'results/chest/THETA{THETA}/lrnr_list-3hcam.pkl', 'wb') as f:
    dill.dump(lrnr_list, f)
with open(f'results/chest/THETA{THETA}/camdata-3hcam.pkl', 'wb') as f:
    dill.dump(camdata, f)