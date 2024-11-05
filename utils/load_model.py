import timm
import torch
from huggingface_hub import login
from torchvision import transforms
from torchvision.transforms import v2
from timm.layers import SwiGLUPacked
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoImageProcessor, AutoModel, ViTModel
from conch.open_clip_custom import create_model_from_pretrained
from .wrapper_model import *
from .model_code.PathDino.PathDino import get_pathDino_model


def load_conch(hf_auth_token=None):
    model, transform = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch",
                                                    hf_auth_token=hf_auth_token)
    model.eval()
    return model, transform, ConchWrapper


def load_uni(hf_auth_token=None):
    if hf_auth_token:
        login(hf_auth_token)  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()
    return model, transform, UniWrapper


def load_pathdino(weights_path):
    model, transform = get_pathDino_model(weights_path=weights_path)
    model.eval()
    return model, transform, PathDinoWrapper


def load_hoptimus0(hf_auth_token=None):
    if hf_auth_token:
        login(hf_auth_token)

    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517)
        ),
    ])
    model.eval()
    return model, transform, Hoptimus0Wrapper


def load_hibou_b(*args):
    transform = AutoImageProcessor.from_pretrained("histai/hibou-B", trust_remote_code=True)
    model = AutoModel.from_pretrained("histai/hibou-B", trust_remote_code=True)
    model.eval()
    return model, transform, HibouWrapper


def load_hibou_l(*args):
    transform = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True)
    model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
    model.eval()
    return model, transform, HibouWrapper


def load_phikon(*args):
    transform = AutoImageProcessor.from_pretrained("owkin/phikon")
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    model.eval()
    return model, transform, PhikonWrapper


def load_prov_gigapath(*args):
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()
    return model, transform, ProvGigaPathWrapper


def load_virchow2(*args):
    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked,
                              act_layer=torch.nn.SiLU)

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()
    return model, transform, Virchow2Wrapper


def load_virchow(*args):
    model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked,
                              act_layer=torch.nn.SiLU)
    model = model.eval()

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()
    return model, transform, VirchowWrapper


def load_kaiko_b(*args):
    model = timm.create_model(
        model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
        dynamic_img_size=True,
        pretrained=True,
    ).eval()

    transform = transforms.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=224),
            v2.CenterCrop(size=224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

    model.eval()
    return model, transform, KaikoWrapper


def load_kaiko_l(*args):
    model = timm.create_model(
        model_name="hf-hub:1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms",
        dynamic_img_size=True,
        pretrained=True,
    ).eval()

    transform = transforms.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=224),
            v2.CenterCrop(size=224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

    model.eval()
    return model, transform, KaikoWrapper


def load_phikon2(*args):
    transform = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2")
    model.eval()
    return model, transform, Phikon2Wrapper


model_registry = {
    None: None,
    'conch': load_conch,
    'uni': load_uni,
    'pathdino': load_pathdino,
    'hibou_l': load_hibou_l,
    'hibou_b': load_hibou_b,
    'phikon': load_phikon,
    'virchow': load_virchow,
    'virchow2': load_virchow2,
    'hoptimus0': load_hoptimus0,
    'kaiko_b': load_kaiko_b,
    'kaiko_l': load_kaiko_l,
    'prov_gigapath': load_prov_gigapath,
    'phikon2': load_phikon2
}
