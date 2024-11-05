import torch


class ModelWrapperBase:
    def __init__(self, model, transform, device):
        self.model = model.to(device)
        self.transform = transform
        self.device = device

    def get_embeddings(self, images):
        raise NotImplementedError("Subclasses must implement `get_embeddings` method.")


class ConchWrapper(ModelWrapperBase):
    def get_embeddings(self, images):
        images = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.inference_mode():
            image_embs = self.model.encode_image(images, proj_contrast=False, normalize=False)
        return image_embs.detach().cpu()


class UniWrapper(ModelWrapperBase):
    def get_embeddings(self, images):
        images = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.inference_mode():
            image_embs = self.model(images)
        return image_embs.detach().cpu()


class PathDinoWrapper(ModelWrapperBase):
    def get_embeddings(self, images):
        images = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            image_embs = self.model(images)
        return image_embs.detach().cpu()


class HibouWrapper(ModelWrapperBase):
    def get_embeddings(self, images):
        inputs = self.transform(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embs = self.model(**inputs)
        return image_embs['pooler_output'].detach().cpu()


class PhikonWrapper(ModelWrapperBase):
    def get_embeddings(self, images):
        inputs = self.transform(images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embs = outputs.last_hidden_state[:, 0, :]
        return image_embs.detach().cpu()


class Virchow2Wrapper(ModelWrapperBase):
    def get_embeddings(self, images):
        images = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = self.model(images)

        class_token = output[:, 0]
        patch_tokens = output[:, 5:]
        image_embs = torch.cat([class_token, patch_tokens.mean(1)], dim=-1).to(torch.float16)
        return image_embs.detach().cpu()


class VirchowWrapper(ModelWrapperBase):
    def get_embeddings(self, images):
        images = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            output = self.model(images)

        class_token = output[:, 0]
        patch_tokens = output[:, 1:]
        image_embs = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return image_embs.detach().cpu()


class Hoptimus0Wrapper(ModelWrapperBase):
    def get_embeddings(self, images):
        images = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            with torch.inference_mode():
                image_embds = self.model(images)
        return image_embds.detach().cpu()


class ProvGigaPathWrapper(ModelWrapperBase):
    def get_embeddings(self, images):
        images = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            image_embds = self.model(images)
        return image_embds.detach().cpu()


class KaikoWrapper(ModelWrapperBase):
    def get_embeddings(self, images):
        images = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            image_embds = self.model(images)
        return image_embds.detach().cpu()


class Phikon2Wrapper(ModelWrapperBase):
    def get_embeddings(self, images):
        inputs = self.transform(images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            image_embds = outputs.last_hidden_state[:, 0, :]
        return image_embds.detach().cpu()
