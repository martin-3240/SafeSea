import argparse
import numpy as np
from PIL import Image

from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch


class BlendedLatnetDiffusion:
    def __init__(self):
        self.parse_args()
        self.load_models()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--prompt", type=str, required=True, help="The target text prompt"
        )
        parser.add_argument(
            "--init_image", type=str, required=True, help="The path to the input image"
        )
        parser.add_argument(
            "--mask", type=str, required=True, help="The path to the input mask"
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default="stabilityai/stable-diffusion-2-1-base",
            help="The path to the HuggingFace model",
        )
        parser.add_argument(
            "--image_id",
            type=str,
            default="",
            help="The path to the HuggingFace model",
        )
        parser.add_argument(
            "--batch_size", type=int, default=4, help="The number of images to generate"
        )
        parser.add_argument(
            "--id", type=int, required=True, help="The target text prompt"
        )
        parser.add_argument(
            "--blending_start_percentage",
            type=float,
            default=0.25,
            help="The diffusion steps percentage to jump",
        )
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument(
            "--output_path",
            type=str,
            default="outputs/res.jpg",
            help="The destination output path",
        )
        parser.add_argument(
            "--neg_prompt", type=str, required=True, help="The target text prompt"
        )
        parser.add_argument(
            "--gen", type=int, default=43, help="The number of images to generate"
        )

        self.args = parser.parse_args()

    def load_models(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.args.model_path, torch_dtype=torch.float16
        )
        self.vae = pipe.vae.to(self.args.device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.args.device)
        self.unet = pipe.unet.to(self.args.device)
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

    #### update
    def get_views(self, panorama_height, panorama_width, window_size=64, stride=8):
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        panorama_height /= 8
        panorama_width /= 8
        num_blocks_height = (panorama_height - window_size) // stride + 1
        num_blocks_width = (panorama_width - window_size) // stride + 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            views.append((h_start, h_end, w_start, w_end))
        return views
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    #### update

    @torch.no_grad()
    def edit_image(
        self,
        image_path,
        mask_path,
        prompts,
        neg_prompts,
        generator,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        blending_percentage=0.25,
    ):
        batch_size = len(prompts)

        image = Image.open(image_path)
        image = image.resize((height, width), Image.BILINEAR)
        image = np.array(image)[:, :, :3]
        source_latents = self._image2latent(image)
        latent_mask, org_mask = self._read_mask(mask_path)

        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to("cuda"))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to("cuda"))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to("cuda").half()

        #### update
        views = self.get_views(height, width)
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)
        #### update

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps[
            int(len(self.scheduler.timesteps) * blending_percentage) :
        ]:
            #### update
            count.zero_()
            value.zero_()
            for h_start, h_end, w_start, w_end in views:
            #### update
                latents_for_view = latents[:, :, h_start:h_end, w_start:w_end]
                    #### Update
                latent_source = source_latents[:, :, h_start:h_end, w_start:w_end]          
                mask_latent = latent_mask[:, :, h_start:h_end, w_start:w_end]
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents_for_view] * 2)

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, timestep=t
                )

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents_view_denoised = self.scheduler.step(noise_pred, t, latents_for_view).prev_sample

                # Blending
                noise_source_latents = self.scheduler.add_noise(
                    source_latents, torch.randn_like(latents_view_denoised), t
                )
                latents_view_denoised = latents_view_denoised * mask_latent + noise_source_latents * (1 - mask_latent)

                #### # noise source latent
                value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                count[:, :, h_start:h_end, w_start:w_end] += 1
            # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
            latents = torch.where(count > 0, value / count, value)

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images

    @torch.no_grad()
    def _image2latent(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to("cuda")
        image = image.half()
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215

        return latents

    def _read_mask(self, mask_path: str, dest_size=(64, 64)):
        org_mask = Image.open(mask_path).convert("L")
        mask = org_mask.resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask[np.newaxis, np.newaxis, ...]
        mask = torch.from_numpy(mask).half().to(self.args.device)

        return mask, org_mask


if __name__ == "__main__":
    bld = BlendedLatnetDiffusion()
    results = bld.edit_image(
        bld.args.init_image,
        bld.args.mask,
        prompts=[bld.args.prompt] * bld.args.batch_size,
        blending_percentage=bld.args.blending_start_percentage,
        neg_prompts = [bld.args.neg_prompt] * bld.args.batch_size,
        generator = torch.manual_seed(bld.args.gen),
    )
    # results_flat = np.concatenate(results, axis=1)
    # Image.fromarray(results_flat).save(bld.args.output_path)
    # for i in range(10):
    #     Image.fromarray(results[i]).save(bld.args.output_path + f"{i+1}.jpg")
    for i in range(10):
        Image.fromarray(results[i]).save(bld.args.output_path + f"{bld.args.gen}_{i+1}_{bld.args.image_id}")

