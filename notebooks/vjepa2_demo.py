# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from transformers import AutoModel, AutoVideoProcessor

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_giant_xformers_rope

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 encoder
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def load_pretrained_vjepa_classifier_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 classifier
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["classifiers"][0]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    # Eval transform has no random cropping nor flip
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return eval_transform


def get_video(video_path="sample_video.mp4", num_frames=64):
    vr = VideoReader(video_path)
    # choosing some frames here, you can define more complex sampling strategy
    total = len(vr)
    step = max(1, total // num_frames)
    frame_idx = np.arange(0, min(total, step * num_frames), step)[:num_frames]
    video = vr.get_batch(frame_idx).asnumpy()
    return video


def visualize_patch_features(features, video_frames, patch_size=16, img_size=384, save_path="feature_viz.gif"):
    """
    Visualize encoder patch features via PCA, saved as a side-by-side GIF.

    Left half:  original video frame
    Right half: top-3 PCA components of the patch features mapped to RGB,
                showing which regions the model represents similarly.

    Args:
        features:     [1, T_eff*H_patches*W_patches, D] tensor (CPU or GPU, any dtype)
        video_frames: [T, C, H, W] uint8 tensor (raw, un-normalized)
        patch_size:   spatial patch size used by the encoder
        img_size:     spatial size the encoder received (after crop/resize transform)
        save_path:    output GIF path
    """
    from PIL import Image

    T, C, H_raw, W_raw = video_frames.shape

    # Patch grid is computed on the transformed (cropped) frame, not the raw frame
    H_patches = img_size // patch_size
    W_patches = img_size // patch_size

    # T_eff may be less than T when tubelet_size > 1 (e.g. tubelet_size=2 halves T)
    N_total = features.shape[1]
    T_eff = N_total // (H_patches * W_patches)

    # Move to CPU float32 numpy: [T_eff*H_patches*W_patches, D]
    feats = features[0].cpu().float().numpy()

    # PCA via SVD (no sklearn needed)
    feats_centered = feats - feats.mean(axis=0)
    _, _, Vt = np.linalg.svd(feats_centered, full_matrices=False)
    pca = feats_centered @ Vt[:3].T  # [T_eff*N_patches, 3]

    # Normalize each component to [0, 1]
    for i in range(3):
        lo, hi = pca[:, i].min(), pca[:, i].max()
        pca[:, i] = (pca[:, i] - lo) / (hi - lo + 1e-8)

    # Reshape to [T_eff, H_patches, W_patches, 3]
    pca = pca.reshape(T_eff, H_patches, W_patches, 3)

    # Upsample temporally to match the number of video frames (nearest-neighbour)
    if T_eff < T:
        indices = (np.arange(T) * T_eff / T).astype(int)
        pca = pca[indices]  # [T, H_patches, W_patches, 3]

    gif_frames = []
    for t in range(T):
        # Original frame: [H_raw, W_raw, C] uint8
        orig = video_frames[t].permute(1, 2, 0).numpy()

        # PCA frame: upsample from patch grid to raw frame dimensions
        pca_img = Image.fromarray((pca[t] * 255).astype(np.uint8), mode="RGB")
        pca_img = pca_img.resize((W_raw, H_raw), Image.BILINEAR)
        pca_arr = np.array(pca_img)

        # Side-by-side: [H_raw, 2*W_raw, 3]
        combined = np.concatenate([orig, pca_arr], axis=1)
        gif_frames.append(Image.fromarray(combined))

    gif_frames[0].save(
        save_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=100,  # ms per frame
        loop=0,
    )
    print(f"Saved feature visualization to {save_path}")


def clear_device_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def forward_vjepa_video(model_hf, model_pt, hf_transform, pt_transform, device, num_frames=64, video_path="sample_video.mp4"):
    # Run a sample inference with VJEPA.
    # Models are run sequentially to keep peak memory low.
    raw = get_video(video_path=video_path, num_frames=num_frames)  # T x H x W x C  (uint8)
    video = torch.from_numpy(raw).permute(0, 3, 1, 2)  # T x C x H x W

    # PT model inference
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.float16):
        x_pt = pt_transform(video).to(device).unsqueeze(0)  # [1, C, T, H, W]
        out_patch_features_pt = model_pt(x_pt)

    clear_device_cache(device)

    # HF model inference
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.float16):
        x_hf = hf_transform(video, return_tensors="pt")["pixel_values_videos"].to(device)
        out_patch_features_hf = model_hf.get_vision_features(x_hf)

    clear_device_cache(device)

    return out_patch_features_hf, out_patch_features_pt, video


def get_vjepa_video_classification_results(classifier, out_patch_features_pt, device):
    SOMETHING_SOMETHING_V2_CLASSES = json.load(open("ssv2_classes.json", "r"))

    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.float16):
        out_classifier = classifier(out_patch_features_pt)

    print(f"Classifier output shape: {out_classifier.shape}")

    print("Top 5 predicted class names:")
    top5_indices = out_classifier.topk(5).indices[0]
    top5_probs = F.softmax(out_classifier.topk(5).values[0], dim=0) * 100.0  # convert to percentage
    for idx, prob in zip(top5_indices, top5_probs):
        str_idx = str(idx.item())
        print(f"{SOMETHING_SOMETHING_V2_CLASSES[str_idx]} ({prob}%)")

    return


def run_sample_inference(video_path="sample_video.mp4", num_frames=16, viz_path=None):
    # Select device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if viz_path is None:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        viz_path = f"{stem}_features.gif"
    elif not viz_path.lower().endswith(".gif"):
        viz_path = os.path.splitext(viz_path)[0] + ".gif"
    print(f"Using device: {device}, num_frames: {num_frames}, video: {video_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # HuggingFace model repo name
    hf_model_name = (
        "facebook/vjepa2-vitg-fpc64-384"  # Replace with your favored model, e.g. facebook/vjepa2-vitg-fpc64-384
    )
    # Path to local PyTorch weights
    pt_model_path = "/Users/mmacklem/Documents/repos/vjepa2/checkpoints/vitg-384.pt"

    # Initialize the HuggingFace model, load pretrained weights in float16 to reduce memory
    model_hf = AutoModel.from_pretrained(hf_model_name, torch_dtype=torch.float16)
    model_hf.to(device).eval()

    # Build HuggingFace preprocessing transform
    hf_transform = AutoVideoProcessor.from_pretrained(hf_model_name)
    img_size = hf_transform.crop_size["height"]  # E.g. 384, 256, etc.

    # Initialize the PyTorch model in float16 to reduce memory, load pretrained weights
    model_pt = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=num_frames)
    model_pt.to(device).half().eval()
    load_pretrained_vjepa_pt_weights(model_pt, pt_model_path)

    # Build PyTorch preprocessing transform
    pt_video_transform = build_pt_video_transform(img_size=img_size)

    # Inference on video
    out_patch_features_hf, out_patch_features_pt, video_frames = forward_vjepa_video(
        model_hf, model_pt, hf_transform, pt_video_transform, device, num_frames=num_frames, video_path=video_path
    )

    print(
        f"""
        Inference results on video:
        HuggingFace output shape: {out_patch_features_hf.shape}
        PyTorch output shape:     {out_patch_features_pt.shape}
        Absolute difference sum:  {torch.abs(out_patch_features_pt - out_patch_features_hf).sum():.6f}
        Close: {torch.allclose(out_patch_features_pt, out_patch_features_hf, atol=1e-3, rtol=1e-3)}
        """
    )

    # Visualize patch features via PCA
    visualize_patch_features(out_patch_features_pt, video_frames, patch_size=16, img_size=img_size, save_path=viz_path)

    # Initialize the classifier in float16 to reduce memory
    classifier_model_path = "/Users/mmacklem/Documents/repos/vjepa2/checkpoints/ssv2-vitg-384-64x2x3.pt"
    classifier = (
        AttentiveClassifier(embed_dim=model_pt.embed_dim, num_heads=16, depth=4, num_classes=174)
        .to(device)
        .half()
        .eval()
    )
    load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)

    # Download SSV2 classes if not already present
    ssv2_classes_path = "ssv2_classes.json"
    if not os.path.exists(ssv2_classes_path):
        command = [
            "wget",
            "https://huggingface.co/datasets/huggingface/label-files/resolve/d79675f2d50a7b1ecf98923d42c30526a51818e2/"
            "something-something-v2-id2label.json",
            "-O",
            "ssv2_classes.json",
        ]
        subprocess.run(command)
        print("Downloading SSV2 classes")

    get_vjepa_video_classification_results(classifier, out_patch_features_pt, device)


if __name__ == "__main__":
    import argparse
    # Run with: `python -m notebooks.vjepa2_demo`
    # Use --num_frames to control memory usage (default 16 for MPS; 64 for full accuracy)
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="sample_video.mp4", help="path to input video file")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--viz_path", type=str, default=None, help="output GIF path (default: <video_stem>_features.gif)")
    args = parser.parse_args()
    run_sample_inference(video_path=args.video, num_frames=args.num_frames, viz_path=args.viz_path)
